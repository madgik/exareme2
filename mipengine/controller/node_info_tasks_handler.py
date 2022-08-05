from typing import Dict
from typing import Final

from celery.result import AsyncResult

from mipengine.celery_app_conf import CELERY_APP_QUEUE_MAX_PRIORITY
from mipengine.controller import controller_logger as ctrl_logger
from mipengine.controller.celery_app import CeleryAppFactory
from mipengine.controller.celery_app import CeleryWrapper
from mipengine.node_info_DTOs import NodeInfo
from mipengine.node_tasks_DTOs import CommonDataElements

TASK_SIGNATURES: Final = {
    "get_node_info": "mipengine.node.tasks.common.get_node_info",
    "get_node_datasets_per_data_model": "mipengine.node.tasks.common.get_node_datasets_per_data_model",
    "get_data_model_cdes": "mipengine.node.tasks.common.get_data_model_cdes",
}


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
    ) -> NodeInfo:
        celery_app = self._get_node_celery_app()
        logger = ctrl_logger.get_request_logger(request_id=request_id)
        result = celery_app.get_result(
            async_result=async_result,
            timeout=self._tasks_timeout,
            logger=logger,
        )
        return NodeInfo.parse_raw(result)

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
