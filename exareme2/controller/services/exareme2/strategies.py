from abc import abstractmethod
from typing import List
from typing import Optional

from exareme2 import exareme2_algorithm_classes
from exareme2 import exareme2_algorithm_data_loaders
from exareme2.algorithms.exareme2.algorithm import AlgorithmInitializationParams
from exareme2.algorithms.exareme2.algorithm import Variables
from exareme2.algorithms.exareme2.longitudinal_transformer import (
    InitializationParams as LongitudinalTransformerRunnerInitParams,
)
from exareme2.algorithms.exareme2.longitudinal_transformer import (
    LongitudinalTransformerRunner,
)
from exareme2.algorithms.exareme2.longitudinal_transformer import (
    LongitudinalTransformerRunnerDataLoader,
)
from exareme2.algorithms.specifications import TransformerName
from exareme2.controller.federation_info_logs import log_experiment_execution
from exareme2.controller.services.api.algorithm_request_dtos import AlgorithmRequestDTO
from exareme2.controller.services.exareme2 import Exareme2Controller
from exareme2.controller.services.exareme2.algorithm_flow_engine_interface import (
    CommandIdGenerator,
)
from exareme2.controller.services.exareme2.algorithm_flow_engine_interface import (
    Exareme2AlgorithmFlowEngineInterface,
)
from exareme2.controller.services.exareme2.algorithm_flow_engine_interface import (
    Exareme2AlgorithmFlowEngineInterfaceSingleLocalWorker,
)
from exareme2.controller.services.exareme2.algorithm_flow_engine_interface import (
    InitializationParams as EngineInitParams,
)
from exareme2.controller.services.exareme2.algorithm_threadpool_executor import (
    algorithm_run_in_threadpool,
)
from exareme2.controller.services.exareme2.data_model_views_creator import (
    DataModelViews,
)
from exareme2.controller.services.exareme2.tasks_handler import Exareme2TasksHandler
from exareme2.controller.services.exareme2.workers_federation import WorkersFederation
from exareme2.controller.services.strategy_interface import AlgorithmExecutionStrategyI


class Exareme2AlgorithmExecutionStrategy(AlgorithmExecutionStrategyI):
    _controller: Exareme2Controller
    _local_worker_tasks_handlers: List[Exareme2TasksHandler]
    _global_worker_tasks_handler: Optional[Exareme2TasksHandler]

    def __init__(
        self,
        controller: Exareme2Controller,
        algorithm_name: str,
        algorithm_request_dto: AlgorithmRequestDTO,
    ):
        super().__init__(controller, algorithm_name, algorithm_request_dto)

        self._algorithm_name = algorithm_name
        self._variables = Variables(
            x=sanitize_request_variable(algorithm_request_dto.inputdata.x),
            y=sanitize_request_variable(algorithm_request_dto.inputdata.y),
        )
        self._command_id_generator = CommandIdGenerator()

        # Instantiate a WorkersFederation that will keep track of the local workers that are
        # relevant to the execution, based on the request parameters
        self._workers_federation = WorkersFederation(
            request_id=self._request_id,
            context_id=self._context_id,
            data_model=self._algorithm_request_dto.inputdata.data_model,
            datasets=self._algorithm_request_dto.inputdata.datasets,
            var_filters=self._algorithm_request_dto.inputdata.filters,
            worker_landscape_aggregator=self._controller.worker_landscape_aggregator,
            celery_tasks_timeout=self._controller.task_timeout,
            celery_run_udf_task_timeout=self._controller.run_udf_task_timeout,
            command_id_generator=self._command_id_generator,
            logger=self._logger,
            global_worker_tasks_handler=self._global_worker_tasks_handler,
            local_worker_tasks_handlers=self._local_worker_tasks_handlers,
        )

        # Add the identifier of the execution(context_id), along with the relevant local
        # worker ids, to the cleaner so that whatever database artifacts are created during
        # the execution get dropped at the end of the execution, when not needed anymore
        self._controller.cleaner.add_contextid_for_cleanup(
            self._context_id, self._workers_federation.worker_ids
        )

        # Initialize metadata
        variable_names = (algorithm_request_dto.inputdata.x or []) + (
            algorithm_request_dto.inputdata.y or []
        )
        self._metadata = self._controller.worker_landscape_aggregator.get_metadata(
            data_model=self._algorithm_request_dto.inputdata.data_model,
            variable_names=variable_names,
        )

        self._initialize_algorithm_flow_engine()

    def _initialize_algorithm_flow_engine(self):
        """
        Instantiate an algorithm execution engine, the engine is passed to the
        "Algorithm" implementation and serves as an API for the "Algorithm" code to
        execute celery on workers
        """
        engine_init_params = EngineInitParams(
            smpc_params=self._controller.smpc_params,
            request_id=self._algorithm_request_dto.request_id,
            algo_flags=self._algorithm_request_dto.flags,
        )
        if len(self._workers_federation.workers.local_workers) < 2:
            self._algorithm_flow_engine = (
                Exareme2AlgorithmFlowEngineInterfaceSingleLocalWorker(
                    initialization_params=engine_init_params,
                    command_id_generator=self._command_id_generator,
                    workers=self._workers_federation.workers,
                )
            )
        else:
            self._algorithm_flow_engine = Exareme2AlgorithmFlowEngineInterface(
                initialization_params=engine_init_params,
                command_id_generator=self._command_id_generator,
                workers=self._workers_federation.workers,
            )

    async def execute_algorithm(
        self,
        algorithm,
        data_model_views: DataModelViews,
        metadata,
    ) -> str:
        log_experiment_execution(
            logger=self._logger,
            request_id=self._workers_federation.workers.local_workers[0].request_id,
            context_id=self._workers_federation.workers.local_workers[0].context_id,
            algorithm_name=self._algorithm_name,
            datasets=self._algorithm_request_dto.inputdata.datasets,
            algorithm_parameters=self._algorithm_request_dto.parameters,
            local_worker_ids=[
                worker.worker_id
                for worker in self._workers_federation.workers.local_workers
            ],
        )

        algorithm_result = await algorithm_run_in_threadpool(
            algorithm=algorithm,
            data_model_views=data_model_views,
            metadata=metadata,
            logger=self._logger,
        )

        algorithm_result_json = algorithm_result.json()

        self._logger.info(
            f"Finished execution->  {self._algorithm_name=} with {self._request_id=}"
        )
        self._logger.debug(
            f"Algorithm {self._request_id=} result-> {algorithm_result_json=}"
        )

        return algorithm_result_json

    @abstractmethod
    async def execute(self) -> str:
        pass


def sanitize_request_variable(variable: list):
    if variable:
        return variable
    else:
        return []


class SingleAlgorithmStrategy(Exareme2AlgorithmExecutionStrategy):
    async def execute(self) -> str:
        algorithm_data_loader = exareme2_algorithm_data_loaders[self._algorithm_name](
            variables=self._variables
        )

        init_params = AlgorithmInitializationParams(
            algorithm_name=self._algorithm_name,
            var_filters=self._algorithm_request_dto.inputdata.filters,
            algorithm_parameters=self._algorithm_request_dto.parameters,
            datasets=self._algorithm_request_dto.inputdata.datasets,
        )
        algorithm = exareme2_algorithm_classes[self._algorithm_name](
            initialization_params=init_params,
            data_loader=algorithm_data_loader,
            engine=self._algorithm_flow_engine,
        )

        data_model_views = self._workers_federation.create_data_model_views(
            variable_groups=algorithm_data_loader.get_variable_groups(),
            dropna=algorithm_data_loader.get_dropna(),
            check_min_rows=algorithm_data_loader.get_check_min_rows(),
        )

        return await self.execute_algorithm(algorithm, data_model_views, self._metadata)


class LongitudinalStrategy(Exareme2AlgorithmExecutionStrategy):
    """
    Used for algorithms that first have a transformation step
    """

    async def _execute_transformation(self):
        transformation_data_loader = LongitudinalTransformerRunnerDataLoader(
            variables=self._variables
        )
        transformation_data_model_views = (
            self._workers_federation.create_data_model_views(
                variable_groups=transformation_data_loader.get_variable_groups(),
                dropna=transformation_data_loader.get_dropna(),
                check_min_rows=transformation_data_loader.get_check_min_rows(),
            )
        )

        init_params = LongitudinalTransformerRunnerInitParams(
            datasets=self._algorithm_request_dto.inputdata.datasets,
            var_filters=self._algorithm_request_dto.inputdata.filters,
            algorithm_parameters=self._algorithm_request_dto.preprocessing.get(
                TransformerName.LONGITUDINAL_TRANSFORMER
            ),
        )
        longitudinal_transformer = LongitudinalTransformerRunner(
            initialization_params=init_params,
            data_loader=transformation_data_loader,
            engine=self._algorithm_flow_engine,
        )

        longitudinal_transformation_result = await algorithm_run_in_threadpool(
            algorithm=longitudinal_transformer,
            data_model_views=transformation_data_model_views,
            metadata=self._metadata,
            logger=self._logger,
        )
        data_transformed = longitudinal_transformation_result.data
        metadata = longitudinal_transformation_result.metadata

        return data_transformed, metadata

    async def execute(self):
        (
            transformation_data,
            transformation_metadata,
        ) = await self._execute_transformation()

        X = transformation_data[0]
        y = transformation_data[1]
        alg_vars = Variables(x=X.columns, y=y.columns)
        algorithm_data_loader = exareme2_algorithm_data_loaders[self._algorithm_name](
            variables=alg_vars
        )

        new_data_model_views = DataModelViews(transformation_data)

        init_params = AlgorithmInitializationParams(
            algorithm_name=self._algorithm_name,
            var_filters=self._algorithm_request_dto.inputdata.filters,
            algorithm_parameters=self._algorithm_request_dto.parameters,
            datasets=self._algorithm_request_dto.inputdata.datasets,
        )
        algorithm = exareme2_algorithm_classes[self._algorithm_name](
            initialization_params=init_params,
            data_loader=algorithm_data_loader,
            engine=self._algorithm_flow_engine,
        )

        return await self.execute_algorithm(
            algorithm, new_data_model_views, transformation_metadata
        )
