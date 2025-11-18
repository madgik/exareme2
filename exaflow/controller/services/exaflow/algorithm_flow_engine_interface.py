from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import List

from exaflow.algorithms.exaflow.exaflow_registry import exaflow_registry
from exaflow.algorithms.exaflow.exaflow_registry import get_udf_registry_key
from exaflow.controller import logger as ctrl_logger
from exaflow.controller.services.exaflow.tasks_handler import ExaflowTasksHandler


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
    ) -> None:
        self._logger = ctrl_logger.get_request_logger(request_id=request_id)
        self._context_id = context_id
        self._tasks_handlers = tasks_handlers

    def run_algorithm_udf(self, func, positional_args) -> List[dict]:
        if not self._tasks_handlers:
            return []

        results: List[dict] = [None] * len(self._tasks_handlers)
        udf_registry_key = get_udf_registry_key(func)
        if "metadata" in positional_args:
            positional_args["metadata"] = add_ordered_enums(positional_args["metadata"])

        with ThreadPoolExecutor(max_workers=len(self._tasks_handlers)) as executor:
            future_to_index = {
                executor.submit(
                    task_handler.run_udf,
                    udf_registry_key=udf_registry_key,
                    params=positional_args,
                ): idx
                for idx, task_handler in enumerate(self._tasks_handlers)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                results[idx] = future.result()
        return results
