import asyncio
from typing import Dict
from typing import List

from exareme2 import flower_algorithm_folder_paths
from exareme2.controller import config as ctrl_config
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.federation_info_logs import log_experiment_execution
from exareme2.controller.services.flower.tasks_handler import FlowerTasksHandler
from exareme2.controller.uid_generator import UIDGenerator
from exareme2.worker_communication import WorkerInfo

FLOWER_SERVER_PORT = "8080"


class WorkerException(Exception):
    pass


class WorkerUnresponsiveException(WorkerException):
    def __init__(self):
        super().__init__("One of the workers stopped responding")


class WorkerTaskTimeoutException(WorkerException):
    def __init__(self, timeout):
        super().__init__(
            f"Task took longer than {timeout} seconds. Increase timeout or try again."
        )


# Controller class
class Controller:
    def __init__(
        self, worker_landscape_aggregator, flower_execution_info, task_timeout
    ):
        self.worker_landscape_aggregator = worker_landscape_aggregator
        self.flower_execution_info = flower_execution_info
        self.task_timeout = task_timeout
        self.lock = asyncio.Lock()

    def _create_worker_tasks_handler(self, request_id, worker_info: WorkerInfo):
        worker_addr = f"{worker_info.ip}:{worker_info.port}"
        worker_db_addr = f"{worker_info.db_ip}:{worker_info.db_port}"
        return FlowerTasksHandler(
            request_id,
            worker_id=worker_info.id,
            worker_queue_addr=worker_addr,
            worker_db_addr=worker_db_addr,
            tasks_timeout=self.task_timeout,
        )

    async def exec_algorithm(self, algorithm_name, algorithm_request_dto):
        async with (self.lock):
            request_id = algorithm_request_dto.request_id
            context_id = UIDGenerator().get_a_uid()
            logger = ctrl_logger.get_request_logger(request_id)
            datasets = algorithm_request_dto.inputdata.datasets + (
                algorithm_request_dto.inputdata.validation_datasets
                if algorithm_request_dto.inputdata.validation_datasets
                else []
            )
            csv_paths_per_worker_id: Dict[
                str, List[str]
            ] = self.worker_landscape_aggregator.get_csv_paths_per_worker_id(
                algorithm_request_dto.inputdata.data_model, datasets
            )

            workers_info = [
                self.worker_landscape_aggregator.get_worker_info(worker_id)
                for worker_id in csv_paths_per_worker_id
            ]
            task_handlers = [
                self._create_worker_tasks_handler(request_id, worker)
                for worker in workers_info
            ]

            global_worker = self.worker_landscape_aggregator.get_global_worker()
            server_task_handler = self._create_worker_tasks_handler(
                request_id, global_worker
            )
            server_ip = global_worker.ip
            server_id = global_worker.id
            # Garbage Collect
            server_task_handler.garbage_collect()
            for handler in task_handlers:
                handler.garbage_collect()

            self.flower_execution_info.set_inputdata(
                inputdata=algorithm_request_dto.inputdata
            )
            server_pid = None
            clients_pids = {}
            server_address = f"{server_ip}:{FLOWER_SERVER_PORT}"
            algorithm_folder_path = flower_algorithm_folder_paths[algorithm_name]
            try:
                server_pid = server_task_handler.start_flower_server(
                    algorithm_folder_path,
                    len(task_handlers),
                    str(server_address),
                    csv_paths_per_worker_id[server_id]
                    if algorithm_request_dto.inputdata.validation_datasets
                    else [],
                )
                clients_pids = {
                    handler.start_flower_client(
                        algorithm_folder_path,
                        str(server_address),
                        csv_paths_per_worker_id[handler.worker_id],
                        ctrl_config.flower_execution_timeout,
                    ): handler
                    for handler in task_handlers
                }

                log_experiment_execution(
                    logger,
                    request_id,
                    context_id,
                    algorithm_name,
                    algorithm_request_dto.inputdata.datasets,
                    algorithm_request_dto.parameters,
                    [info.id for info in workers_info],
                )
                result = await self.flower_execution_info.get_result_with_timeout()

                logger.info(f"Finished execution -> {algorithm_name} with {request_id}")
                return result

            except asyncio.TimeoutError:
                raise WorkerTaskTimeoutException(self.task_timeout)
            finally:
                await self._cleanup(
                    algorithm_name, server_task_handler, server_pid, clients_pids
                )

    def _create_global_handler(self, request_id):
        global_worker = self.worker_landscape_aggregator.get_global_worker()
        return (
            self._create_worker_tasks_handler(request_id, global_worker),
            global_worker.ip,
        )

    async def _cleanup(
        self, algorithm_name, server_task_handler, server_pid, clients_pids
    ):
        await self.flower_execution_info.reset()
        server_task_handler.stop_flower_server(server_pid, algorithm_name)
        for pid, handler in clients_pids.items():
            handler.stop_flower_client(pid, algorithm_name)
