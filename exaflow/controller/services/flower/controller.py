import asyncio

from exaflow.controller import config as ctrl_config
from exaflow.controller import logger as ctrl_logger
from exaflow.controller.services.controller_interface import ControllerI
from exaflow.controller.services.flower.flower_io_registry import FlowerIORegistry
from exaflow.controller.services.flower.tasks_handler import FlowerTasksHandler
from exaflow.worker_communication import WorkerInfo


class FlowerController(ControllerI):
    def __init__(
        self,
        worker_landscape_aggregator,
        task_timeout,
    ):
        super().__init__(worker_landscape_aggregator, task_timeout)

        self.flower_execution_info = FlowerIORegistry(
            ctrl_config.flower.execution_timeout,
            ctrl_logger.get_background_service_logger(),
        )
        self.algorithm_execution_lock = asyncio.Lock()

    def create_worker_tasks_handler(
        self, request_id, worker_info: WorkerInfo
    ) -> FlowerTasksHandler:
        worker_addr = f"{worker_info.ip}:{worker_info.port}"
        return FlowerTasksHandler(
            request_id,
            worker_id=worker_info.id,
            worker_queue_addr=worker_addr,
            worker_db_addr=worker_db_addr,
            tasks_timeout=self.task_timeout,
        )
