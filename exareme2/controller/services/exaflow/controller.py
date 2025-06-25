from exareme2.controller import logger as ctrl_logger
from exareme2.controller.services.exaflow.tasks_handler import TasksHandler
from exareme2.controller.uid_generator import UIDGenerator


class Controller:
    def __init__(self, worker_landscape_aggregator, task_timeout: int) -> None:
        self.worker_landscape_aggregator = worker_landscape_aggregator
        self.task_timeout = task_timeout

    def _create_worker_tasks_handler(
        self, request_id: str, worker_info
    ) -> TasksHandler:
        return TasksHandler(
            request_id=request_id,
            worker_id=worker_info.id,
            worker_queue_addr=f"{worker_info.ip}:{worker_info.port}",
            tasks_timeout=self.task_timeout,
        )

    async def exec_algorithm(
        self,
        algorithm_name: str,
        algorithm_request_dto,
        strategy,  # instance of ControllerExecutionStrategy
    ):
        request_id = algorithm_request_dto.request_id
        context_id = UIDGenerator().get_a_uid()
        logger = ctrl_logger.get_request_logger(request_id)

        datasets = algorithm_request_dto.inputdata.datasets
        worker_ids = (
            self.worker_landscape_aggregator.get_worker_ids_with_any_of_datasets(
                algorithm_request_dto.inputdata.data_model,
                datasets,
            )
        )
        workers_info = [
            self.worker_landscape_aggregator.get_worker_info(w_id)
            for w_id in worker_ids
        ]

        task_handlers = [
            self._create_worker_tasks_handler(request_id, info) for info in workers_info
        ]

        variable_names = (algorithm_request_dto.inputdata.x or []) + (
            algorithm_request_dto.inputdata.y or []
        )
        metadata = self.worker_landscape_aggregator.get_metadata(
            data_model=algorithm_request_dto.inputdata.data_model,
            variable_names=variable_names,
        )

        return await strategy.execute(
            request_id,
            context_id,
            algorithm_name,
            algorithm_request_dto,
            task_handlers,
            metadata,
            logger,
        )
