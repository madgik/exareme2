from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import List

from exaflow.algorithms.exareme3.exareme3_registry import get_udf_registry_key
from exaflow.controller import logger as ctrl_logger
from exaflow.controller.services.exareme3.tasks_handler import ExaflowTasksHandler
from exaflow.worker_communication import InsufficientDataError


def add_ordered_enums(data_dict):
    for key, field in data_dict.items():
        if field.get("is_categorical") and "enumerations" in field:
            # extract the codes (keys of the enumerations dict)
            ordered = list(field["enumerations"].keys())
            # add to the same dictionary
            field["ordered_enums"] = ordered
    return data_dict


class ExaflowAlgorithmFlowEngineInterface:
    def __init__(
        self,
        request_id: str,
        context_id: str,
        tasks_handlers: List[ExaflowTasksHandler],
        preprocessing=None,
        raw_inputdata=None,
    ) -> None:
        self._logger = ctrl_logger.get_request_logger(request_id=request_id)
        self._context_id = context_id
        self._tasks_handlers = tasks_handlers
        self._preprocessing = preprocessing
        self._raw_inputdata = raw_inputdata

    def run_algorithm_udf(self, func, positional_args) -> List[dict]:
        if not self._tasks_handlers:
            return []

        results: List[dict] = [None] * len(self._tasks_handlers)
        udf_registry_key = get_udf_registry_key(func)
        if "metadata" in positional_args:
            positional_args["metadata"] = add_ordered_enums(positional_args["metadata"])

        params = dict(positional_args)
        if self._preprocessing:
            params["preprocessing"] = self._preprocessing
        if self._raw_inputdata:
            params["raw_inputdata"] = self._raw_inputdata.json()

        executor = ThreadPoolExecutor(max_workers=len(self._tasks_handlers))
        future_to_index = {}
        pending = set()
        try:
            future_to_index = {
                executor.submit(
                    task_handler.run_udf,
                    udf_registry_key=udf_registry_key,
                    params=params,
                ): idx
                for idx, task_handler in enumerate(self._tasks_handlers)
            }
            pending = set(future_to_index)
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except InsufficientDataError as exc:
                    self._logger.warning(
                        "Worker %s returned insufficient data and will be skipped: %s",
                        self._tasks_handlers[idx].worker_id,
                        exc,
                    )
                    results[idx] = None
                pending.discard(future)
        except Exception:
            for future in future_to_index:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=not pending, cancel_futures=bool(pending))
        filtered_results = [res for res in results if res is not None]
        if not filtered_results:
            raise InsufficientDataError(
                "No workers had sufficient data to run the requested algorithm."
            )
        return filtered_results
