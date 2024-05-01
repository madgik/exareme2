import asyncio
from typing import List

from exareme2.controller import logger as ctrl_logger
from exareme2.controller.federation_info_logs import log_experiment_execution
from exareme2.controller.services.flower.tasks_handler import FlowerTasksHandler
from exareme2.controller.uid_generator import UIDGenerator
from exareme2.worker_communication import WorkerInfo


# Base Exception class for Worker-related exceptions
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
        async with self.lock:
            request_id = algorithm_request_dto.request_id
            context_id = UIDGenerator().get_a_uid()
            logger = ctrl_logger.get_request_logger(request_id)
            workers_info = self._get_workers_info_by_dataset(
                algorithm_request_dto.inputdata.data_model,
                algorithm_request_dto.inputdata.datasets,
            )
            task_handlers = [
                self._create_worker_tasks_handler(request_id, worker)
                for worker in workers_info
            ]
            server_task_handler = (
                task_handlers[0]
                if len(task_handlers) == 1
                else self._create_global_handler(request_id)
            )
            self.flower_execution_info.set_inputdata(
                inputdata=algorithm_request_dto.inputdata
            )
            server_pid = None
            clients_pids = {}

            try:
                server_pid = server_task_handler.start_flower_server(
                    algorithm_name, len(task_handlers)
                )
                clients_pids = {
                    handler.start_flower_client(algorithm_name): handler
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
                result = await self.flower_execution_info.get_result_with_timeout(
                    self.task_timeout
                )

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
        return self._create_worker_tasks_handler(request_id, global_worker)

    async def _cleanup(
        self, algorithm_name, server_task_handler, server_pid, clients_pids
    ):
        await self.flower_execution_info.reset()
        server_task_handler.stop_flower_server(server_pid, algorithm_name)
        for pid, handler in clients_pids.items():
            handler.stop_flower_client(pid, algorithm_name)

    def _get_workers_info_by_dataset(self, data_model, datasets) -> List[WorkerInfo]:
        """Retrieves worker information for those handling the specified datasets."""
        worker_ids = (
            self.worker_landscape_aggregator.get_worker_ids_with_any_of_datasets(
                data_model, datasets
            )
        )
        return [
            self.worker_landscape_aggregator.get_worker_info(worker_id)
            for worker_id in worker_ids
        ]
