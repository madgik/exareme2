from typing import Dict

from exareme2.controller import logger as ctrl_logger
from exareme2.controller.celery.tasks_handler import WorkerTasksHandler
from exareme2.worker_communication import CommonDataElements
from exareme2.worker_communication import DataModelAttributes
from exareme2.worker_communication import DatasetsInfoPerDataModel
from exareme2.worker_communication import WorkerInfo


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

    def get_worker_datasets_per_data_model_task(self) -> DatasetsInfoPerDataModel:
        result = self._worker_tasks_handler.queue_worker_datasets_per_data_model_task(
            self._request_id
        ).get(self._tasks_timeout)
        return DatasetsInfoPerDataModel.parse_raw(result)

    def get_data_model_cdes_task(self, data_model: str) -> CommonDataElements:
        result = self._worker_tasks_handler.queue_data_model_cdes_task(
            request_id=self._request_id,
            data_model=data_model,
        ).get(self._tasks_timeout)
        return CommonDataElements.parse_raw(result)

    def get_data_model_attributes_task(self, data_model: str) -> DataModelAttributes:
        result = self._worker_tasks_handler.queue_data_model_attributes_task(
            self._request_id, data_model
        ).get(self._tasks_timeout)
        return DataModelAttributes.parse_raw(result)

    def get_healthcheck_task(self, check_db: bool):
        return self._worker_tasks_handler.queue_healthcheck_task(
            request_id=self._request_id,
            check_db=check_db,
        ).get(self._tasks_timeout)

    def load_worker_data(self, folder: str | None = None):
        return self._worker_tasks_handler.queue_load_data_folder_task(
            request_id=self._request_id,
            folder=folder,
        ).get(self._tasks_timeout)
