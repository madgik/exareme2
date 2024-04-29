import asyncio
from logging import Logger
from typing import Dict
from typing import List

from exareme2.controller.services.flower.execution_engine import (
    AlgorithmExecutionEngine,
)
from exareme2.controller.services.flower.execution_engine import Workers
from exareme2.controller.services.flower.tasks_handler import FlowerTasksHandler
from exareme2.controller.services.flower.workers import GlobalWorker
from exareme2.controller.services.flower.workers import LocalWorker
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    WorkerLandscapeAggregator,
)
from exareme2.controller.uid_generator import UIDGenerator
from exareme2.worker_communication import WorkerInfo


class WorkerUnresponsiveException(Exception):
    """Exception raised when a worker stops responding during algorithm execution."""

    def __init__(self):
        super().__init__("One of the workers stopped responding")


class WorkerTaskTimeoutException(Exception):
    """Exception for task timeout, potentially caused by high load or large data."""

    def __init__(self, timeout):
        super().__init__(
            f"Task took longer than {timeout} seconds. Increase timeout or try again."
        )


class WorkersFederation:
    """
    Manages a federation of workers, selecting those that contain the relevant data for algorithm execution.
    It initializes workers based on the data model and datasets constraints provided.
    """

    def __init__(
        self,
        request_id: str,
        context_id: str,
        data_model: str,
        datasets: List[str],
        var_filters: Dict,
        worker_aggregator: WorkerLandscapeAggregator,
        task_timeout: int,
        logger: Logger,
    ):
        self._logger = logger
        self._request_id = request_id
        self._context_id = context_id
        self._data_model = data_model
        self._datasets = datasets
        self._var_filters = var_filters
        self._worker_aggregator = worker_aggregator
        self._task_timeout = task_timeout
        self._workers = self._initialize_workers()

    def _initialize_workers(self) -> Workers:
        local_workers_info = self._get_workers_info_by_dataset()
        local_workers = self._create_local_workers(local_workers_info)

        # Check if there is only one local worker; if so, it acts as the global worker.
        global_worker = self._create_global_worker(
            local_workers_info[0]
            if len(local_workers) == 1
            else self._worker_aggregator.get_global_worker()
        )

        return Workers(global_worker=global_worker, local_workers=local_workers)

    def _get_workers_info_by_dataset(self) -> List[WorkerInfo]:
        """Retrieves worker information for those handling the specified datasets."""
        worker_ids = self._worker_aggregator.get_worker_ids_with_any_of_datasets(
            self._data_model, self._datasets
        )
        return [
            self._worker_aggregator.get_worker_info(worker_id)
            for worker_id in worker_ids
        ]

    def _create_local_workers(
        self, worker_info_list: List[WorkerInfo]
    ) -> List[LocalWorker]:
        return [
            LocalWorker(
                self._request_id,
                self._context_id,
                self._create_worker_tasks_handler(worker_info),
                self._data_model,
                self._get_datasets_for_worker(worker_info.id),
            )
            for worker_info in worker_info_list
        ]

    def _get_datasets_for_worker(self, worker_id: str) -> List[str]:
        """Determines which datasets a worker will handle based on their capabilities and the current data model."""
        return self._worker_aggregator.get_worker_specific_datasets(
            worker_id, self._data_model, self._datasets
        )

    def _create_worker_tasks_handler(
        self, worker_info: WorkerInfo
    ) -> FlowerTasksHandler:
        return FlowerTasksHandler(
            request_id=self._request_id,
            worker_id=worker_info.id,
            worker_queue_addr=f"{worker_info.ip}:{worker_info.port}",
            worker_db_addr=f"{worker_info.db_ip}:{worker_info.db_port}",
            tasks_timeout=self._task_timeout,
        )

    def _create_global_worker(self, worker_info: WorkerInfo) -> GlobalWorker:
        tasks_handler = self._create_worker_tasks_handler(worker_info)
        return GlobalWorker(self._request_id, self._context_id, tasks_handler)

    @property
    def active_workers(self) -> Workers:
        return self._workers


class AlgorithmExecutor:
    def __init__(self, engine, flower_experiment_watcher, algorithm_name, logger):
        self._engine = engine
        self._algorithm_name = algorithm_name
        self._flower_experiment_watcher = flower_experiment_watcher
        self._logger = logger

    async def run(self, tasks_timeout):
        process_id_per_worker = self._engine.start_flower(self._algorithm_name)
        try:
            algorithm_result = (
                await self._flower_experiment_watcher.get_result_with_timeout(
                    tasks_timeout
                )
            )
            return algorithm_result
        except asyncio.TimeoutError:
            error_msg = f"Timeout retrieving algorithm result within {tasks_timeout} seconds for {self._algorithm_name}."
            self._logger.error(error_msg)
            raise TimeoutError(error_msg) from None
        except Exception as exc:
            error_msg = f"Unexpected error during algorithm execution: {exc}"
            self._logger.error(error_msg)
            raise RuntimeError(error_msg) from exc
        finally:
            try:
                await self._flower_experiment_watcher.reset()
            finally:
                self._engine.stop_flower(process_id_per_worker, self._algorithm_name)


class Controller:
    def __init__(
        self, worker_landscape_aggregator, flower_execition_info, logger, task_timeout
    ):
        self._logger = logger
        self._worker_landscape_aggregator = worker_landscape_aggregator
        self._flower_execition_info = flower_execition_info
        self._task_timeout = task_timeout

    async def exec_algorithm(self, algorithm_name, algorithm_request_dto):
        request_id = algorithm_request_dto.request_id
        context_id = UIDGenerator().get_a_uid()
        try:
            workers_federation = WorkersFederation(
                request_id,
                context_id,
                algorithm_request_dto.inputdata.data_model,
                algorithm_request_dto.inputdata.datasets,
                algorithm_request_dto.inputdata.filters,
                self._worker_landscape_aggregator,
                self._task_timeout,
                self._logger,
            )

            engine = AlgorithmExecutionEngine(
                request_id, workers_federation.active_workers
            )
            self._flower_execition_info.set_inputdata(algorithm_request_dto.inputdata)

            executor = AlgorithmExecutor(
                engine, self._flower_execition_info, algorithm_name, self._logger
            )
            result = await executor.run(self._task_timeout)
            self._logger.info(
                f"Finished execution -> {algorithm_name} with {request_id}"
            )
            return result
        except Exception as exc:
            self._logger.error(
                f"Failed to execute {algorithm_name} with {request_id}: {exc}"
            )
            raise
