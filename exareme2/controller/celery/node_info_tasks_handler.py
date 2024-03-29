from typing import Dict
from typing import Final

from celery.result import AsyncResult

from exareme2.celery_app_conf import CELERY_APP_QUEUE_MAX_PRIORITY
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.celery.app import CeleryAppFactory
from exareme2.controller.celery.app import CeleryWrapper
from exareme2.worker_communication import CommonDataElements
from exareme2.worker_communication import DataModelAttributes
from exareme2.worker_communication import WorkerInfo

TASK_SIGNATURES: Final = {
    "get_node_info": "exareme2.worker.worker_info.worker_info_api.get_worker_info",
    "get_node_datasets_per_data_model": "exareme2.worker.worker_info.worker_info_api.get_node_datasets_per_data_model",
    "get_data_model_cdes": "exareme2.worker.worker_info.worker_info_api.get_data_model_cdes",
    "get_data_model_attributes": "exareme2.worker.worker_info.worker_info_api.get_data_model_attributes",
    "healthcheck": "exareme2.worker.worker_info.worker_info_api.healthcheck",
}


# TODO (Refactor) Split the task handlers from the celery logic
# The interface should be used in the engines and celery/grpc should implement them.
# The interface task handler should be in the services package.
class NodeInfoTasksHandler:
    def __init__(self, node_queue_addr: str, tasks_timeout: int):
        self._node_queue_addr = node_queue_addr
        self._tasks_timeout = tasks_timeout

    @property
    def node_queue_addr(self) -> str:
        return self._node_queue_addr

    def _get_node_celery_app(self) -> CeleryWrapper:
        return CeleryAppFactory().get_celery_app(socket_addr=self._node_queue_addr)

    # --------------- get_node_info task ---------------
    # NON-BLOCKING
    def queue_node_info_task(self, request_id: str) -> AsyncResult:
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["get_node_info"]
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=request_id,
            priority=CELERY_APP_QUEUE_MAX_PRIORITY,
        )
        return async_result

    # BLOCKING
    def result_node_info_task(
        self, async_result: AsyncResult, request_id: str
    ) -> WorkerInfo:
        celery_app = self._get_node_celery_app()
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            logger=logger,
        )
        return WorkerInfo.parse_raw(result)

    # --------------- get_node_datasets_per_data_model task ---------------
    # NON-BLOCKING
    def queue_node_datasets_per_data_model_task(self, request_id: str) -> AsyncResult:
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["get_node_datasets_per_data_model"]
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=request_id,
            priority=CELERY_APP_QUEUE_MAX_PRIORITY,
        )
        return async_result

    # BLOCKING
    def result_node_datasets_per_data_model_task(
        self, async_result: AsyncResult, request_id: str
    ) -> Dict[str, Dict[str, str]]:
        celery_app = self._get_node_celery_app()
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            logger=logger,
        )
        return result

    # --------------- get_data_model_cdes task ---------------
    # NON-BLOCKING
    def queue_data_model_cdes_task(
        self, request_id: str, data_model: str
    ) -> AsyncResult:
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["get_data_model_cdes"]
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=request_id,
            data_model=data_model,
            priority=CELERY_APP_QUEUE_MAX_PRIORITY,
        )
        return async_result

    # BLOCKING
    def result_data_model_cdes_task(
        self, async_result: AsyncResult, request_id: str
    ) -> CommonDataElements:
        celery_app = self._get_node_celery_app()
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            logger=logger,
        )
        return CommonDataElements.parse_raw(result)

    # --------------- get_data_model_attributes task ---------------
    # NON-BLOCKING
    def queue_data_model_attributes_task(
        self, request_id: str, data_model: str
    ) -> AsyncResult:
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["get_data_model_attributes"]
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=request_id,
            data_model=data_model,
            priority=CELERY_APP_QUEUE_MAX_PRIORITY,
        )
        return async_result

    # BLOCKING
    def result_data_model_attributes_task(
        self, async_result: AsyncResult, request_id: str
    ) -> DataModelAttributes:
        celery_app = self._get_node_celery_app()
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            logger=logger,
        )
        return DataModelAttributes.parse_raw(result)

    # --------------- healthcheck task ---------------
    # NON-BLOCKING
    def queue_healthcheck_task(self, request_id: str, check_db: bool) -> AsyncResult:
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["healthcheck"]
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        async_result = celery_app.queue_task(
            task_signature=task_signature,
            logger=logger,
            request_id=request_id,
            check_db=check_db,
            priority=CELERY_APP_QUEUE_MAX_PRIORITY,
        )
        return async_result

    # BLOCKING
    def result_healthcheck(self, async_result: AsyncResult, request_id: str):
        celery_app = self._get_node_celery_app()
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        return celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            logger=logger,
        )
