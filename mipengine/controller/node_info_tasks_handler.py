from typing import Callable
from typing import Final
from typing import List
from typing import Optional
from typing import Tuple

import kombu
from billiard.exceptions import SoftTimeLimitExceeded
from billiard.exceptions import TimeLimitExceeded
from celery.exceptions import TimeoutError
from celery.result import AsyncResult
from kombu.exceptions import OperationalError

# from mipengine.controller.celery_app import get_node_celery_app
from mipengine.controller.celery_app import CeleryAppFactory
from mipengine.controller.celery_app import CeleryConnectionError
from mipengine.controller.celery_app import CeleryTaskTimeoutException
from mipengine.controller.node_tasks_handler_interface import INodeTasksHandler
from mipengine.controller.node_tasks_handler_interface import IQueuedUDFAsyncResult
from mipengine.controller.node_tasks_handler_interface import UDFKeyArguments
from mipengine.controller.node_tasks_handler_interface import UDFPosArguments
from mipengine.node_info_DTOs import NodeInfo
from mipengine.node_tasks_DTOs import CommonDataElements
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import UDFResults

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

    # from multiprocessing.pool import ThreadPool
    # from multiprocessing.pool import AsyncResult as threading_AsyncResult
    # def result_node_info_task_async(
    #     self, async_result: AsyncResult
    # ) -> threading_AsyncResult:

    #     pool = ThreadPool()
    #     threading_async_result = pool.apply_async(
    #         self.result_node_info_task,
    #         kwds={
    #             "async_result": async_result,
    #         },
    #     )
    #     return threading_async_result

    def result_node_info_task(self, async_result: AsyncResult) -> NodeInfo:
        celery_app = self._get_node_celery_app()
        try:
            result = celery_app.get_result(
                async_result=async_result, timeout=self._tasks_timeout
            )
            return NodeInfo.parse_raw(result)
        except (CeleryTaskTimeoutException, ConnectionError) as exc:
            # TODO: how should log here???
            print(f"{exc=}")
            raise exc

    # --------------- get_node_datasets_per_data_model task ---------------
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

    # def result_node_datasets_per_data_model_task_async(
    #     self, async_result: AsyncResult
    # ) -> threading_AsyncResult:
    #     from multiprocessing.pool import ThreadPool

    #     pool = ThreadPool()
    #     threading_async_result = pool.apply_async(
    #         self.result_node_datasets_per_data_model_task,
    #         kwds={
    #             "async_result": async_result,
    #         },
    #     )
    #     return threading_async_result

    def result_node_datasets_per_data_model_task(
        self, async_result: AsyncResult
    ) -> NodeInfo:
        celery_app = self._get_node_celery_app()
        try:
            result = celery_app.get_result(
                async_result=async_result, timeout=self._tasks_timeout
            )
            return result
        except (CeleryTaskTimeoutException, CeleryConnectionError) as exc:
            # TODO: how should log here???
            print(f"{exc=}")
            raise exc

    # --------------- get_data_model_cdes task ---------------
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

    # def result_data_model_cdes_task_async(
    #     self, async_result: AsyncResult
    # ) -> threading_AsyncResult:
    #     from multiprocessing.pool import ThreadPool

    #     pool = ThreadPool()
    #     threading_async_result = pool.apply_async(
    #         self.result_data_model_cdes_task,
    #         kwds={
    #             "async_result": async_result,
    #         },
    #     )
    #     return threading_async_result

    def result_data_model_cdes_task(self, async_result: AsyncResult) -> NodeInfo:
        celery_app = self._get_node_celery_app()
        try:
            result = celery_app.get_result(
                async_result=async_result, timeout=self._tasks_timeout
            )
            return CommonDataElements.parse_raw(result)
        except (CeleryTaskTimeoutException, CeleryConnectionError) as exc:
            # TODO: how should log here???
            print(f"{exc=}")
            raise exc
