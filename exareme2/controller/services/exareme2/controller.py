from logging import Logger
from typing import Optional

from exareme2.algorithms.exareme2.algorithm import Variables
from exareme2.algorithms.specifications import TransformerName
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.services.api.algorithm_request_dtos import AlgorithmRequestDTO
from exareme2.controller.services.controller_interface import ControllerI
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
from exareme2.controller.services.exareme2.algorithm_flow_engine_interface import (
    SMPCParams,
)
from exareme2.controller.services.exareme2.algorithm_flow_engine_interface import (
    Workers,
)
from exareme2.controller.services.exareme2.cleaner import Cleaner
from exareme2.controller.services.exareme2.strategies import LongitudinalStrategy
from exareme2.controller.services.exareme2.strategies import SingleAlgorithmStrategy
from exareme2.controller.services.exareme2.tasks_handler import Exareme2TasksHandler
from exareme2.controller.services.exareme2.workers_federation import WorkersFederation
from exareme2.controller.services.strategy_interface import AlgorithmExecutionStrategyI
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    WorkerLandscapeAggregator,
)
from exareme2.controller.uid_generator import UIDGenerator
from exareme2.worker_communication import WorkerInfo


class Exareme2Controller(ControllerI):
    cleaner: Cleaner
    run_udf_task_timeout: int
    smpc_params: SMPCParams

    def __init__(
        self,
        worker_landscape_aggregator: WorkerLandscapeAggregator,
        cleaner: Cleaner,
        logger: Logger,
        task_timeout: int,
        run_udf_task_timeout: int,
        smpc_params: SMPCParams,
    ):
        super().__init__(worker_landscape_aggregator, task_timeout)

        self._controller_logger = logger
        self.cleaner = cleaner
        self.run_udf_task_timeout = run_udf_task_timeout
        self.smpc_params = smpc_params

    def create_worker_tasks_handler(
        self,
        request_id: str,
        worker_info: WorkerInfo,
    ) -> Exareme2TasksHandler:
        return Exareme2TasksHandler(
            request_id=request_id,
            worker_id=worker_info.id,
            worker_queue_addr=str(worker_info.ip) + ":" + str(worker_info.port),
            worker_db_addr=str(worker_info.db_ip) + ":" + str(worker_info.db_port),
            tasks_timeout=self.task_timeout,
            run_udf_task_timeout=self.run_udf_task_timeout,
        )

    def start_cleanup_loop(self):
        self._controller_logger.info("(Controller) Cleaner starting ...")
        self.cleaner.start()
        self._controller_logger.info("(Controller) Cleaner started.")

    def stop_cleanup_loop(self):
        self.cleaner.stop()

    async def exec_algorithm(
        self,
        algorithm_name: str,
        algorithm_request_dto: AlgorithmRequestDTO,
        strategy: Optional[AlgorithmExecutionStrategyI] = None,
    ) -> str:
        command_id_generator = CommandIdGenerator()

        request_id = algorithm_request_dto.request_id
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        context_id = UIDGenerator().get_a_uid()
        data_model = algorithm_request_dto.inputdata.data_model
        datasets = algorithm_request_dto.inputdata.datasets
        var_filters = algorithm_request_dto.inputdata.filters

        # Instantiate a WorkersFederation that will keep track of the local workers that are
        # relevant to the execution, based on the request parameters
        workers_federation = WorkersFederation(
            request_id=request_id,
            context_id=context_id,
            data_model=data_model,
            datasets=datasets,
            var_filters=var_filters,
            worker_landscape_aggregator=self.worker_landscape_aggregator,
            celery_tasks_timeout=self.task_timeout,
            celery_run_udf_task_timeout=self.run_udf_task_timeout,
            command_id_generator=command_id_generator,
            logger=logger,
            global_worker_tasks_handler=self.get_global_worker_tasks_handler(
                request_id
            ),
            local_worker_tasks_handlers=self.get_local_worker_tasks_handlers(
                data_model, datasets, request_id
            ),
        )

        # add the identifier of the execution(context_id), along with the relevant local
        # worker ids, to the cleaner so that whatever database artifacts are created during
        # the execution get dropped at the end of the execution, when not needed anymore
        self.cleaner.add_contextid_for_cleanup(
            context_id, workers_federation.worker_ids
        )

        # get metadata
        variable_names = (algorithm_request_dto.inputdata.x or []) + (
            algorithm_request_dto.inputdata.y or []
        )
        metadata = self.worker_landscape_aggregator.get_metadata(
            data_model=data_model, variable_names=variable_names
        )

        # instantiate an algorithm execution engine, the engine is passed to the
        # "Algorithm" implementation and serves as an API for the "Algorithm" code to
        # execute celery on workers
        engine_init_params = EngineInitParams(
            smpc_params=self.smpc_params,
            request_id=algorithm_request_dto.request_id,
            algo_flags=algorithm_request_dto.flags,
        )
        engine = _create_algorithm_execution_engine(
            engine_init_params=engine_init_params,
            command_id_generator=command_id_generator,
            workers=workers_federation.workers,
        )
        variables = Variables(
            x=sanitize_request_variable(algorithm_request_dto.inputdata.x),
            y=sanitize_request_variable(algorithm_request_dto.inputdata.y),
        )

        # Choose ExecutionStrategy
        if (
            algorithm_request_dto.preprocessing
            and algorithm_request_dto.preprocessing.get(
                TransformerName.LONGITUDINAL_TRANSFORMER
            )
        ):
            execution_strategy = LongitudinalStrategy(
                algorithm_name=algorithm_name,
                variables=variables,
                algorithm_request_dto=algorithm_request_dto,
                engine=engine,
                logger=logger,
            )
        else:
            execution_strategy = SingleAlgorithmStrategy(
                algorithm_name=algorithm_name,
                variables=variables,
                algorithm_request_dto=algorithm_request_dto,
                engine=engine,
                logger=logger,
            )

        # Create the "data model views"
        data_model_views = workers_federation.create_data_model_views(
            variable_groups=execution_strategy.algorithm_data_loader.get_variable_groups(),
            dropna=execution_strategy.algorithm_data_loader.get_dropna(),
            check_min_rows=execution_strategy.algorithm_data_loader.get_check_min_rows(),
        )

        # Execute the strategy
        algorithm_result = await execution_strategy.run(
            data=data_model_views, metadata=metadata
        )

        logger.info(f"Finished execution->  {algorithm_name=} with {request_id=}")
        logger.debug(f"Algorithm {request_id=} result-> {algorithm_result=}")

        # Cleanup artifacts created in the workers' databases during the execution
        if not self.cleaner.cleanup_context_id(context_id=context_id):
            # if the cleanup did not succeed, set the current "context_id" as released
            # so that the Cleaner retries later
            self.cleaner.release_context_id(context_id=context_id)

        return algorithm_result


def _create_algorithm_execution_engine(
    engine_init_params: EngineInitParams,
    command_id_generator: CommandIdGenerator,
    workers: Workers,
):
    if len(workers.local_workers) < 2:
        return Exareme2AlgorithmFlowEngineInterfaceSingleLocalWorker(
            initialization_params=engine_init_params,
            command_id_generator=command_id_generator,
            workers=workers,
        )
    else:
        return Exareme2AlgorithmFlowEngineInterface(
            initialization_params=engine_init_params,
            command_id_generator=command_id_generator,
            workers=workers,
        )


def sanitize_request_variable(variable: list):
    if variable:
        return variable
    else:
        return []
