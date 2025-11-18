from exaflow.controller.services.controller_interface import ControllerI
from exaflow.controller.services.exaflow.tasks_handler import ExaflowTasksHandler


class ExaflowController(ControllerI):
    def __init__(self, worker_landscape_aggregator, task_timeout: int) -> None:
        super().__init__(worker_landscape_aggregator, task_timeout)

    def create_worker_tasks_handler(
        self, request_id: str, worker_info
    ) -> ExaflowTasksHandler:
        return ExaflowTasksHandler(
            request_id=request_id,
            worker_id=worker_info.id,
            worker_queue_addr=f"{worker_info.ip}:{worker_info.port}",
            tasks_timeout=self.task_timeout,
        )
