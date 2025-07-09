from typing import List

from exareme2.algorithms.exaflow.exaflow_registry import exaflow_registry
from exareme2.algorithms.exaflow.exaflow_registry import get_udf_registry_key
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.services.exaflow.tasks_handler import ExaflowTasksHandler


class ExaflowAlgorithmFlowEngineInterface:
    def __init__(
        self,
        request_id: str,
        context_id: str,
        tasks_handlers: List[ExaflowTasksHandler],
    ) -> None:
        self._logger = ctrl_logger.get_request_logger(request_id=request_id)
        self._context_id = context_id
        self._tasks_handlers = tasks_handlers

    def run_algorithm_udf(self, func, positional_args) -> List[dict]:
        tasks = []
        for task_handler in self._tasks_handlers:
            udf_registry_key = get_udf_registry_key(func)
            task = task_handler.queue_udf(
                udf_registry_key=udf_registry_key, params=positional_args
            )
            tasks.append((task, task_handler.tasks_timeout))

        return [task.get(timeout) for task, timeout in tasks]
