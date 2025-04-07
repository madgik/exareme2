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
        tasks_handlers: List[TasksHandler],
    ):
        self._logger = ctrl_logger.get_request_logger(request_id=request_id)
        self._request_id = request_id
        self._context_id = context_id
        self._tasks_handlers = tasks_handlers

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
            # Queue the UDF and store both the task and its timeout
            task = task_handler.queue_udf(udf_name=func, params=positional_args)
            tasks.append((task, task_handler.tasks_timeout))

        # Directly call .get(timeout) on each task result
        return [task.get(timeout) for task, timeout in tasks]

    def run_algorithm_udf_with_aggregator(self, func, positional_args) -> dict:
        """
        Executes the given UDF on all local workers, verifies that all returned results are identical,
        and returns the first result.

        The function queues the UDF on each worker and collects the resulting task objects along with
        their respective timeout values. It then retrieves the results using each task's `.get(timeout)` method.
        After all results are obtained, it iterates through them to ensure they are all equal. If they are,
        the first result is returned. Otherwise, a ValueError is raised.

        Parameters:
            func: The UDF to execute.
            positional_args (dict): The arguments for the UDF.

        Returns:
            dict: The result of the UDF execution (all worker results are expected to be identical).

        Raises:
            ValueError: If the results from the workers do not match.
        """
        tasks = []
        for task_handler in self._tasks_handlers:
            # Ensure the request_id is included in the positional arguments.
            positional_args["request_id"] = self._request_id
            task = task_handler.queue_udf(udf_name=func, params=positional_args)
            tasks.append((task, task_handler.tasks_timeout))

        # Retrieve results from all tasks.
        results = [task.get(timeout) for task, timeout in tasks]

        # Verify that all worker results are identical.
        first_result = results[0]
        for result in results:
            if result != first_result:
                raise ValueError(
                    "Worker results do not match: found differing results."
                )

        return first_result
