from logging import Logger
from typing import Final
from typing import List
from typing import Optional

from celery.result import AsyncResult

from exareme2.celery_app_conf import CELERY_APP_QUEUE_MAX_PRIORITY
from exareme2.controller.celery.app import CeleryAppFactory
from exareme2.controller.celery.app import CeleryWrapper
from exareme2.worker_communication import TableInfo
from exareme2.worker_communication import TableSchema
from exareme2.worker_communication import WorkerUDFKeyArguments
from exareme2.worker_communication import WorkerUDFPosArguments

TASK_SIGNATURES: Final = {
    "get_worker_info": "exareme2.worker.worker_info.worker_info_api.get_worker_info",
    "get_worker_datasets_per_data_model": "exareme2.worker.worker_info.worker_info_api.get_worker_datasets_per_data_model",
    "get_data_model_cdes": "exareme2.worker.worker_info.worker_info_api.get_data_model_cdes",
    "get_data_model_attributes": "exareme2.worker.worker_info.worker_info_api.get_data_model_attributes",
    "healthcheck": "exareme2.worker.worker_info.worker_info_api.healthcheck",
    "start_flower_client": "exareme2.worker.flower.starter.starter_api.start_flower_client",
    "start_flower_server": "exareme2.worker.flower.starter.starter_api.start_flower_server",
    "stop_flower_server": "exareme2.worker.flower.cleanup.cleanup_api.stop_flower_server",
    "stop_flower_client": "exareme2.worker.flower.cleanup.cleanup_api.stop_flower_client",
    "garbage_collect": "exareme2.worker.flower.cleanup.cleanup_api.garbage_collect",
    "run_exaflow_udf": "exareme2.worker.exaflow.udf.udf_api.run_udf",
    "load_data_folder": "exareme2.worker.data_management.data_loader_api.load_data_folder",
}


class WorkerTaskResult:
    def __init__(
        self, celery_app: CeleryWrapper, async_result: AsyncResult, logger: Logger
    ):
        self._celery_app = celery_app
        self._async_result = async_result
        self._logger = logger

    def get(self, timeout: int):
        return self._celery_app.get_result(
            async_result=self._async_result, timeout=timeout, logger=self._logger
        )


class WorkerTasksHandler:
    def __init__(self, worker_queue_addr: str, logger: Logger):
        self._worker_queue_addr = worker_queue_addr
        self._logger = logger

    def _get_celery_app(self):
        return CeleryAppFactory().get_celery_app(socket_addr=self._worker_queue_addr)

    def _queue_task(self, task_signature: str, **task_params) -> WorkerTaskResult:
        """
        Queues a task with the given signature and parameters.

        :param task_signature: The signature of the task to queue.
        :param task_params: A dictionary of parameters to pass to the task.
        :return: A WorkerTaskResult instance.
        """
        async_result = self._get_celery_app().queue_task(
            task_signature=task_signature,
            logger=self._logger,
            **task_params,
        )
        return WorkerTaskResult(self._get_celery_app(), async_result, self._logger)

    def queue_worker_info_task(self, request_id: str) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["get_worker_info"],
            request_id=request_id,
            priority=CELERY_APP_QUEUE_MAX_PRIORITY,
        )

    def queue_worker_datasets_per_data_model_task(
        self, request_id: str
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["get_worker_datasets_per_data_model"],
            request_id=request_id,
            priority=CELERY_APP_QUEUE_MAX_PRIORITY,
        )

    def queue_data_model_cdes_task(
        self, request_id: str, data_model: str
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["get_data_model_cdes"],
            request_id=request_id,
            data_model=data_model,
            priority=CELERY_APP_QUEUE_MAX_PRIORITY,
        )

    def queue_data_model_attributes_task(
        self, request_id: str, data_model: str
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["get_data_model_attributes"],
            request_id=request_id,
            data_model=data_model,
            priority=CELERY_APP_QUEUE_MAX_PRIORITY,
        )

    # --------------- healthcheck task ---------------
    # NON-BLOCKING
    def queue_healthcheck_task(
        self, request_id: str, check_db: bool
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["healthcheck"],
            request_id=request_id,
            check_db=check_db,
            priority=CELERY_APP_QUEUE_MAX_PRIORITY,
        )

    def start_flower_client(
        self,
        request_id,
        algorithm_folder_path,
        server_address,
        data_model,
        datasets,
        execution_timeout,
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["start_flower_client"],
            request_id=request_id,
            algorithm_folder_path=algorithm_folder_path,
            server_address=server_address,
            data_model=data_model,
            datasets=datasets,
            execution_timeout=execution_timeout,
        )

    def start_flower_server(
        self,
        request_id,
        algorithm_folder_path,
        number_of_clients,
        server_address,
        data_model,
        datasets,
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["start_flower_server"],
            request_id=request_id,
            algorithm_folder_path=algorithm_folder_path,
            number_of_clients=number_of_clients,
            server_address=server_address,
            data_model=data_model,
            datasets=datasets,
        )

    def stop_flower_server(
        self, request_id, pid: int, algorithm_name: str
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["stop_flower_server"],
            request_id=request_id,
            pid=pid,
            algorithm_name=algorithm_name,
        )

    def stop_flower_client(
        self, request_id, pid: int, algorithm_name: str
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["stop_flower_client"],
            request_id=request_id,
            pid=pid,
            algorithm_name=algorithm_name,
        )

    def garbage_collect(self, request_id) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["garbage_collect"],
            request_id=request_id,
        )

    def queue_udf(
        self,
        request_id,
        udf_registry_key,
        params,
    ) -> WorkerTaskResult:
        return self._queue_task(
            task_signature=TASK_SIGNATURES["run_exaflow_udf"],
            request_id=request_id,
            udf_registry_key=udf_registry_key,
            params=params,
        )

    def queue_load_data_folder_task(
        self, request_id: str, folder: Optional[str] = None
    ) -> WorkerTaskResult:
        task_params = {
            "task_signature": TASK_SIGNATURES["load_data_folder"],
            "request_id": request_id,
            "priority": CELERY_APP_QUEUE_MAX_PRIORITY,
        }
        if folder:
            task_params["folder"] = folder
        return self._queue_task(**task_params)
