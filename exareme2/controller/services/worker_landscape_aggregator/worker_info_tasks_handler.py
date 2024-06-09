import json

from exareme2.controller import logger as ctrl_logger
from exareme2.controller.celery.tasks_handler import WorkerTasksHandler
from exareme2.worker_communication import WorkerInfo
from exareme2.worker_communication import parse_data_model_metadata


class WorkerInfoTasksHandler:
    def __init__(self, worker_queue_addr: str, tasks_timeout: int, request_id: str):
        self._worker_queue_addr = worker_queue_addr
        self._tasks_timeout = tasks_timeout
        self._request_id = request_id
        self._logger = ctrl_logger.get_request_logger(request_id=request_id)
        self._worker_tasks_handler = WorkerTasksHandler(
            self._worker_queue_addr, self._logger
        )

    def get_worker_info_task(self) -> WorkerInfo:
        result = self._worker_tasks_handler.queue_worker_info_task(
            self._request_id
        ).get(self._tasks_timeout)
        return WorkerInfo.parse_raw(result)

    def get_worker_data_model_metadata_and_datasets(self):
        (
            data_models_metadata,
            datasets_per_data_model,
        ) = self._worker_tasks_handler.queue_worker_data_model_metadata_and_dataset_locations(
            self._request_id
        ).get(
            self._tasks_timeout
        )
        return {
            data_model: parse_data_model_metadata(metadata)
            for data_model, metadata in data_models_metadata.items()
        }, datasets_per_data_model

    def get_healthcheck_task(self, check_db: bool):
        return self._worker_tasks_handler.queue_healthcheck_task(
            request_id=self._request_id,
            check_db=check_db,
        ).get(self._tasks_timeout)
