from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

from exareme2.algorithms.exaflow.exaflow_registry import exaflow_registry
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.services.exaflow.tasks_handler import TasksHandler


class AlgorithmExecutionEngine:
    """
    Helper handed to algorithms so they can send UDFs to the workers.

    Notes
    -----
    * `run_algorithm_udf`  – fire-and-collect.
    * `run_algorithm_udf_with_aggregator` – same, but requires **identical**
      results from every worker (often used when a global aggregation server is in play).
    """

    def __init__(
        self,
        request_id: str,
        context_id: str,
        tasks_handlers: List[TasksHandler],
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
        """Internal helper shared by the two public methods below."""
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

    # ------------------------------------------------------------------ #
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
