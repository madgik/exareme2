import asyncio
import concurrent
import traceback
from abc import abstractmethod
from logging import Logger
from typing import List

from exareme2 import AlgorithmDataLoader
from exareme2 import exareme2_algorithm_classes
from exareme2 import exareme2_algorithm_data_loaders
from exareme2.algorithms.exareme2.algorithm import (
    InitializationParams as AlgorithmInitParams,
)
from exareme2.algorithms.exareme2.algorithm import Variables
from exareme2.algorithms.exareme2.longitudinal_transformer import (
    DataLoader as LongitudinalTransformerRunnerDataLoader,
)
from exareme2.algorithms.exareme2.longitudinal_transformer import (
    InitializationParams as LongitudinalTransformerRunnerInitParams,
)
from exareme2.algorithms.exareme2.longitudinal_transformer import (
    LongitudinalTransformerRunner,
)
from exareme2.algorithms.specifications import TransformerName
from exareme2.controller.celery.app import CeleryConnectionError
from exareme2.controller.celery.app import CeleryTaskTimeoutException
from exareme2.controller.federation_info_logs import log_experiment_execution
from exareme2.controller.services.api.algorithm_request_dtos import AlgorithmRequestDTO
from exareme2.controller.services.exareme2.algorithm_flow_engine_interface import (
    Exareme2AlgorithmFlowEngineInterface,
)
from exareme2.controller.services.exareme2.data_model_views_creator import (
    DataModelViews,
)


class WorkerUnresponsiveException(Exception):
    def __init__(self):
        message = (
            "One of the workers participating in the algorithm execution "
            "stopped responding"
        )
        super().__init__(message)
        self.message = message


class WorkerTaskTimeoutException(Exception):
    def __init__(self):
        message = (
            f"One of the celery in the algorithm execution took longer to finish than "
            f"the timeout.This could be caused by a high load or by an experiment with "
            f"too much data. Please try again or increase the timeout."
        )
        super().__init__(message)
        self.message = message


class Exareme2AlgorithmExecutionStrategy:
    def __init__(
        self,
        algorithm_name: str,
        variables: Variables,
        algorithm_request_dto: AlgorithmRequestDTO,
        engine: Exareme2AlgorithmFlowEngineInterface,
        logger: Logger,
    ):
        self._algorithm_name = algorithm_name
        self._variables = variables
        self._algorithm_data_loader = exareme2_algorithm_data_loaders[algorithm_name](
            variables=variables
        )
        self._algorithm_request_dto = algorithm_request_dto
        self._engine = engine
        self._logger = logger

    @property
    def algorithm_data_loader(self):
        return self._algorithm_data_loader

    @abstractmethod
    async def run(self, data, metadata):
        pass

    async def execute(self) -> str:
        pass


class LongitudinalStrategy(Exareme2AlgorithmExecutionStrategy):
    """
    Implementation of ExecutionStrategy interface that first executes
    "LongitudinalTransformer" and then the requested "Algorithm".
    """

    def __init__(
        self,
        algorithm_name: str,
        variables: Variables,
        algorithm_request_dto: AlgorithmRequestDTO,
        engine: Exareme2AlgorithmFlowEngineInterface,
        logger: Logger,
    ):
        super().__init__(
            algorithm_name=algorithm_name,
            variables=variables,
            algorithm_request_dto=algorithm_request_dto,
            engine=engine,
            logger=logger,
        )

        self._algorithm_data_loader = LongitudinalTransformerRunnerDataLoader(
            variables=variables
        )

    async def run(self, data, metadata):
        init_params = LongitudinalTransformerRunnerInitParams(
            datasets=self._algorithm_request_dto.inputdata.datasets,
            var_filters=self._algorithm_request_dto.inputdata.filters,
            algorithm_parameters=self._algorithm_request_dto.preprocessing.get(
                TransformerName.LONGITUDINAL_TRANSFORMER
            ),
        )
        longitudinal_transformer = LongitudinalTransformerRunner(
            initialization_params=init_params,
            data_loader=self._algorithm_data_loader,
            engine=self._engine,
        )

        longitudinal_transform_result = await _algorithm_run_in_event_loop(
            algorithm=longitudinal_transformer,
            data_model_views=data,
            metadata=metadata,
        )
        data_transformed = longitudinal_transform_result.data
        metadata = longitudinal_transform_result.metadata

        X = data_transformed[0]
        y = data_transformed[1]
        alg_vars = Variables(x=X.columns, y=y.columns)
        algorithm_data_loader = exareme2_algorithm_data_loaders[self._algorithm_name](
            variables=alg_vars
        )

        new_data_model_views = DataModelViews(data_transformed)

        algorithm_executor = AlgorithmExecutor(
            engine=self._engine,
            algorithm_data_loader=algorithm_data_loader,
            algorithm_name=self._algorithm_name,
            datasets=self._algorithm_request_dto.inputdata.datasets,
            filters=self._algorithm_request_dto.inputdata.filters,
            params=self._algorithm_request_dto.parameters,
            logger=self._logger,
        )

        algorithm_result = await algorithm_executor.run(
            data=new_data_model_views, metadata=metadata
        )
        return algorithm_result


class SingleAlgorithmStrategy(Exareme2AlgorithmExecutionStrategy):
    """
    Implementation of ExecutionStrategy interface that executes the requested
    "Algorithm" without any other preprocessing steps.
    """

    def __init__(
        self,
        algorithm_name: str,
        variables: Variables,
        algorithm_request_dto: AlgorithmRequestDTO,
        engine: Exareme2AlgorithmFlowEngineInterface,
        logger: Logger,
    ):
        super().__init__(
            algorithm_name=algorithm_name,
            variables=variables,
            algorithm_request_dto=algorithm_request_dto,
            engine=engine,
            logger=logger,
        )

    async def run(self, data, metadata):
        algorithm_executor = AlgorithmExecutor(
            engine=self._engine,
            algorithm_data_loader=self._algorithm_data_loader,
            algorithm_name=self._algorithm_name,
            datasets=self._algorithm_request_dto.inputdata.datasets,
            filters=self._algorithm_request_dto.inputdata.filters,
            params=self._algorithm_request_dto.parameters,
            logger=self._logger,
        )

        algorithm_result = await algorithm_executor.run(data=data, metadata=metadata)
        return algorithm_result


class AlgorithmExecutor:
    """
    Implements the functionality of executing one "Algorithm" asynchronously
    (which must be the final step of any ExecutionStrategy)
    """

    def __init__(
        self,
        engine: Exareme2AlgorithmFlowEngineInterface,
        algorithm_data_loader: AlgorithmDataLoader,
        algorithm_name: str,
        datasets: List[str],
        filters: dict,
        params: dict,
        logger: Logger,
    ):
        self._engine = engine

        self._algorithm_data_loader = algorithm_data_loader

        self._algorithm_name = algorithm_name
        self._datasets = datasets
        self._filters = filters
        self._params = params

        self._logger = logger

    async def run(self, data, metadata):
        # Instantiate Algorithm object
        init_params = AlgorithmInitParams(
            algorithm_name=self._algorithm_name,
            var_filters=self._filters,
            algorithm_parameters=self._params,
            datasets=self._datasets,
        )
        algorithm = exareme2_algorithm_classes[self._algorithm_name](
            initialization_params=init_params,
            data_loader=self._algorithm_data_loader,
            engine=self._engine,
        )

        log_experiment_execution(
            logger=self._logger,
            request_id=self._engine._workers.local_workers[0].request_id,
            context_id=self._engine._workers.local_workers[0].context_id,
            algorithm_name=self._algorithm_name,
            datasets=self._datasets,
            algorithm_parameters=self._params,
            local_worker_ids=[
                worker.worker_id for worker in self._engine._workers.local_workers
            ],
        )

        # Call Algorithm.run inside event loop
        try:
            algorithm_result = await _algorithm_run_in_event_loop(
                algorithm=algorithm,
                data_model_views=data,
                metadata=metadata,
            )
        except CeleryConnectionError as exc:
            self._logger.error(f"ErrorType: '{type(exc)}' and message: '{exc}'")
            raise WorkerUnresponsiveException()
        except CeleryTaskTimeoutException as exc:
            self._logger.error(f"ErrorType: '{type(exc)}' and message: '{exc}'")
            raise WorkerTaskTimeoutException()
        except Exception as exc:
            self._logger.error(traceback.format_exc())
            raise exc

        return algorithm_result.json()


_thread_pool_executor = concurrent.futures.ThreadPoolExecutor()


# TODO add types
# TODO change func name, Transformers runs through this as well
async def _algorithm_run_in_event_loop(algorithm, data_model_views, metadata):
    # By calling blocking method Algorithm.run() inside run_in_executor(),
    # Algorithm.run() will execute in a separate thread of the threadpool and at
    # the same time yield control to the executor event loop, through await
    loop = asyncio.get_event_loop()
    algorithm_result = await loop.run_in_executor(
        _thread_pool_executor,
        algorithm.execute,
        data_model_views.to_list(),
        metadata,
    )
    return algorithm_result
