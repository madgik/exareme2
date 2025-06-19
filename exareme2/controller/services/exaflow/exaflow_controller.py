from __future__ import annotations

from exareme2 import exaflow_algorithm_classes
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.federation_info_logs import log_experiment_execution
from exareme2.controller.services.exaflow.execution_engine import (
    AlgorithmExecutionEngine,
)
from exareme2.controller.services.exaflow.tasks_handler import TasksHandler
from exareme2.controller.uid_generator import UIDGenerator
from exareme2.worker_communication import WorkerInfo


class ExaflowController:
    """
    Base controller coordinating a single algorithm run across the federation.

    Subclasses can override aggregation client setup/teardown if needed.
    """

    def __init__(self, worker_landscape_aggregator, task_timeout: int) -> None:
        self.worker_landscape_aggregator = worker_landscape_aggregator
        self.task_timeout = task_timeout

    def _create_worker_tasks_handler(self, request_id: str, worker_info: WorkerInfo):
        """Factory for TasksHandler so the call site stays concise."""
        return TasksHandler(
            request_id=request_id,
            worker_id=worker_info.id,
            worker_queue_addr=f"{worker_info.ip}:{worker_info.port}",
            tasks_timeout=self.task_timeout,
        )

    def _configure_aggregator(
        self, request_id: str, workers_info: list[WorkerInfo]
    ) -> object | None:
        """
        Hook to configure an aggregation client. Base version does nothing.
        Returns an aggregation client or None.
        """
        return None

    def _cleanup_aggregator(self, agg_client: object) -> None:
        """
        Hook to clean up aggregation client. Base version does nothing.
        """
        pass

    async def exec_algorithm(self, algorithm_name: str, algorithm_request_dto):
        request_id = algorithm_request_dto.request_id
        context_id = UIDGenerator().get_a_uid()
        logger = ctrl_logger.get_request_logger(request_id)

        # 1. Pick workers
        datasets = algorithm_request_dto.inputdata.datasets
        worker_ids = (
            self.worker_landscape_aggregator.get_worker_ids_with_any_of_datasets(
                algorithm_request_dto.inputdata.data_model, datasets
            )
        )
        workers_info = [
            self.worker_landscape_aggregator.get_worker_info(w_id)
            for w_id in worker_ids
        ]

        # 2. Aggregation Client setup (noop in base)
        agg_client = self._configure_aggregator(request_id, workers_info)

        try:
            # 3. Build execution engine
            task_handlers = [
                self._create_worker_tasks_handler(request_id, info)
                for info in workers_info
            ]
            engine = AlgorithmExecutionEngine(
                request_id=request_id,
                context_id=context_id,
                tasks_handlers=task_handlers,
            )

            # 4. Instantiate and run algorithm
            algorithm_cls = exaflow_algorithm_classes[algorithm_name]
            algorithm = algorithm_cls(
                inputdata=algorithm_request_dto.inputdata, engine=engine
            )

            variable_names = (algorithm_request_dto.inputdata.x or []) + (
                algorithm_request_dto.inputdata.y or []
            )
            metadata = self.worker_landscape_aggregator.get_metadata(
                data_model=algorithm_request_dto.inputdata.data_model,
                variable_names=variable_names,
            )

            log_experiment_execution(
                logger,
                request_id,
                context_id,
                algorithm_name,
                datasets,
                algorithm_request_dto.parameters,
                [info.id for info in workers_info],
            )

            result = algorithm.run(metadata=metadata)
            logger.info(f"Finished execution -> {algorithm_name} with {request_id}")
            return result.json()

        finally:
            # 5. Aggregation Client teardown (noop in base)
            if agg_client:
                try:
                    self._cleanup_aggregator(agg_client)
                except Exception:
                    logger = ctrl_logger.get_request_logger(request_id)
                    logger.warning(
                        "Failed to clean up aggregation client", exc_info=True
                    )
