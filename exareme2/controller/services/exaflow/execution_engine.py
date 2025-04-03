from typing import List

from exareme2.controller import logger as ctrl_logger
from exareme2.controller.services.exaflow.tasks_handler import TasksHandler


class AlgorithmExecutionEngine:
    """
    The AlgorithmExecutionEngine is the class used by the algorithms to communicate with
    the workers of the system. An AlgorithmExecutionEngine object is passed to all algorithms
    """

    def __init__(
        self,
        request_id: str,
        context_id: str,
        csv_paths_per_worker_id: dict,
        tasks_handlers: List[TasksHandler],
    ):
        self._logger = ctrl_logger.get_request_logger(request_id=request_id)
        self._context_id = context_id
        self._tasks_handlers = tasks_handlers
        self._csv_paths_per_worker_id = csv_paths_per_worker_id

    def run_algorithm_udf(self, func, positional_args) -> List[dict]:
        """
        Executes the given UDF on all local workers and returns the results.

        This version queues the UDF on each worker and collects the resulting task objects along
        with their respective timeout values. The results are then retrieved by directly calling the
        task's `.get(timeout)` method, eliminating the need for an extra wrapper method.

        Parameters:
            func: The UDF to execute.
            positional_args (dict): The arguments for the UDF.

        Returns:
            List[dict]: A list of results from each worker.
        """
        tasks = []
        for task_handler in self._tasks_handlers:
            # Copy the positional arguments to avoid side effects
            current_positional_args = positional_args.copy()
            # Inject the worker-specific CSV paths
            current_positional_args["csv_paths"] = self._csv_paths_per_worker_id[
                task_handler.worker_id
            ]
            # Queue the UDF and store both the task and its timeout
            task = task_handler.queue_udf(udf_name=func, params=current_positional_args)
            tasks.append((task, task_handler.tasks_timeout))

        # Directly call .get(timeout) on each task result
        return [task.get(timeout) for task, timeout in tasks]
