from typing import Final

from celery.result import AsyncResult

from mipengine.controller.celery_app import CeleryAppFactory
from mipengine.controller.celery_app import CeleryConnectionError
from mipengine.controller.celery_app import CeleryTaskTimeoutException
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

    def _get_node_celery_app(self):
        return CeleryAppFactory().get_celery_app(socket_addr=self._node_queue_addr)

    # --------------- get_node_info task ---------------
    # NON-BLOCKING
    def queue_node_info_task(self, request_id: str) -> AsyncResult:
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["get_node_info"]
        try:
            async_result = celery_app.queue_task(
                task_signature=task_signature,
                request_id=request_id,
            )
            return async_result
        except CeleryConnectionError as exc:
            # TODO: how should log here???
            print(f"{exc=}")
            raise exc

    # BLOCKING
    def result_node_info_task(
        self, async_result: AsyncResult, request_id: str
    ) -> NodeInfo:
        celery_app = self._get_node_celery_app()
        try:
            result = celery_app.get_result(
                async_result=async_result,
                timeout=self._tasks_timeout,
                request_id=request_id,
            )
            return NodeInfo.parse_raw(result)
        except (CeleryTaskTimeoutException, ConnectionError) as exc:
            # TODO: how should log here???
            print(f"{exc=}")
            raise exc

    # --------------- get_node_datasets_per_data_model task ---------------
    # NON-BLOCKING
    def queue_node_datasets_per_data_model_task(self, request_id: str) -> AsyncResult:
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["get_node_datasets_per_data_model"]
        try:
            async_result = celery_app.queue_task(
                task_signature=task_signature,
                request_id=request_id,
            )
            return async_result
        except CeleryConnectionError as exc:
            # TODO: how should log here???
            print(f"{exc=}")
            raise exc

    # BLOCKING
    def result_node_datasets_per_data_model_task(
        self, async_result: AsyncResult, request_id: str
    ) -> NodeInfo:
        celery_app = self._get_node_celery_app()
        try:
            result = celery_app.get_result(
                async_result=async_result,
                timeout=self._tasks_timeout,
                request_id=request_id,
            )
            return result
        except (CeleryTaskTimeoutException, CeleryConnectionError) as exc:
            # TODO: how should log here???
            print(f"{exc=}")
            raise exc

    # --------------- get_data_model_cdes task ---------------
    # NON-BLOCKING
    def queue_data_model_cdes_task(
        self, request_id: str, data_model: str
    ) -> AsyncResult:
        celery_app = self._get_node_celery_app()
        task_signature = TASK_SIGNATURES["get_data_model_cdes"]
        try:
            async_result = celery_app.queue_task(
                task_signature=task_signature,
                request_id=request_id,
                data_model=data_model,
            )
            return async_result
        except CeleryConnectionError as exc:
            # TODO: how should log here???
            print(f"{exc=}")
            raise exc

    # BLOCKING
    def result_data_model_cdes_task(
        self, async_result: AsyncResult, request_id: str
    ) -> NodeInfo:
        celery_app = self._get_node_celery_app()
        try:
            result = celery_app.get_result(
                async_result=async_result,
                timeout=self._tasks_timeout,
                request_id=request_id,
            )
            return CommonDataElements.parse_raw(result)
        except (CeleryTaskTimeoutException, CeleryConnectionError) as exc:
            # TODO: how should log here???
            print(f"{exc=}")
            raise exc
