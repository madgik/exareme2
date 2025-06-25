from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

from exareme2.algorithms.exaflow.exaflow_registry import exaflow_registry
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.services.exaflow.tasks_handler import ExaflowTasksHandler


class ExaflowAlgorithmFlowEngineInterface:
    """
    Used from the algorithm developer in the algorithm flow to execute tasks in the engine.

    TODO Kostas, refactor the engine interface.
    0) Please.... less chatGPT...
    1) Define worker_tasks_handlers and global_worker_tasks_handler
    2) What does _broadcast_udf mean? Why not a similar name to exareme2?
    3) The engine interface should not be different for aggregation server and without it
    4) We need a simpler way to collect the results from the exaflow with aggregation server algorithms
        Maybe for now, have that logic in the algorithm and we improve later.
    """

    def __init__(
        self,
        request_id: str,
        context_id: str,
        tasks_handlers: List[ExaflowTasksHandler],
    ) -> None:
        self._logger = ctrl_logger.get_request_logger(request_id=request_id)
        self._request_id = request_id
        self._context_id = context_id
        self._tasks_handlers = tasks_handlers

    def _broadcast_udf(
        self,
        func: Union[str, Callable],
        positional_args: Dict[str, Any],
        *,
        use_aggregator: bool = False,
    ):
        key = exaflow_registry.resolve_key(func)
        tasks = [
            (
                handler.queue_udf(
                    udf_name=key,
                    params=dict(positional_args),  # avoid accidental mutation
                    use_aggregator=use_aggregator,
                ),
                handler.tasks_timeout,
            )
            for handler in self._tasks_handlers
        ]
        return [task.get(timeout) for task, timeout in tasks]

    def run_algorithm_udf(
        self, func: Union[str, Callable], positional_args: Dict[str, Any]
    ) -> List[dict]:
        return self._broadcast_udf(func, positional_args)

    def run_algorithm_udf_with_aggregator(
        self, func: Union[str, Callable], positional_args: Dict[str, Any]
    ) -> dict:
        """Same, but insists every worker returns the **same** payload."""
        results = self._broadcast_udf(func, positional_args)
        first = results[0]
        if any(r != first for r in results[1:]):
            raise ValueError("Worker results do not match")
        return first
