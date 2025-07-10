from exareme2.controller import logger as ctrl_logger
from exareme2.controller.celery.tasks_handler import WorkerTaskResult
from exareme2.controller.celery.tasks_handler import WorkerTasksHandler
from exareme2.controller.services.tasks_handler_interface import TasksHandlerI


class ExaflowTasksHandler(TasksHandlerI):
    def __init__(
        self,
        request_id: str,
        worker_id: str,
        worker_queue_addr: str,
        tasks_timeout: int,
    ):
        self._request_id = request_id
        self._worker_id = worker_id
        self._worker_queue_addr = worker_queue_addr
        self._tasks_timeout = tasks_timeout
        self._logger = ctrl_logger.get_request_logger(request_id=request_id)
        self._worker_tasks_handler = WorkerTasksHandler(
            self._worker_queue_addr, self._logger
        )

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def tasks_timeout(self) -> int:
        return self._tasks_timeout

    def queue_udf(self, udf_registry_key, params: dict) -> WorkerTaskResult:
        return self._worker_tasks_handler.queue_udf(
            self._request_id, udf_registry_key, params
        )
