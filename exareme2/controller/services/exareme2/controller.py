from logging import Logger

from exareme2.controller.services.controller_interface import ControllerI
from exareme2.controller.services.exareme2.algorithm_flow_engine_interface import (
    SMPCParams,
)
from exareme2.controller.services.exareme2.cleaner import Cleaner
from exareme2.controller.services.exareme2.tasks_handler import Exareme2TasksHandler
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    WorkerLandscapeAggregator,
)
from exareme2.worker_communication import WorkerInfo


class Exareme2Controller(ControllerI):
    cleaner: Cleaner
    run_udf_task_timeout: int
    smpc_params: SMPCParams

    def __init__(
        self,
        worker_landscape_aggregator: WorkerLandscapeAggregator,
        cleaner: Cleaner,
        logger: Logger,
        task_timeout: int,
        run_udf_task_timeout: int,
        smpc_params: SMPCParams,
    ):
        super().__init__(worker_landscape_aggregator, task_timeout)

        self._controller_logger = logger
        self.cleaner = cleaner
        self.run_udf_task_timeout = run_udf_task_timeout
        self.smpc_params = smpc_params

    def create_worker_tasks_handler(
        self,
        request_id: str,
        worker_info: WorkerInfo,
    ) -> Exareme2TasksHandler:
        return Exareme2TasksHandler(
            request_id=request_id,
            worker_id=worker_info.id,
            worker_queue_addr=str(worker_info.ip) + ":" + str(worker_info.port),
            worker_db_addr=str(worker_info.monetdb_configs.ip)
            + ":"
            + str(worker_info.monetdb_configs.port),
            tasks_timeout=self.task_timeout,
            run_udf_task_timeout=self.run_udf_task_timeout,
        )

    def start_cleanup_loop(self):
        self._controller_logger.info("(Controller) Cleaner starting ...")
        self.cleaner.start()
        self._controller_logger.info("(Controller) Cleaner started.")

    def stop_cleanup_loop(self):
        self.cleaner.stop()
