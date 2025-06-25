import asyncio
from typing import Optional

from exareme2 import flower_algorithm_folder_paths
from exareme2.controller import config as ctrl_config
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.federation_info_logs import log_experiment_execution
from exareme2.controller.services.api.algorithm_request_dtos import AlgorithmRequestDTO
from exareme2.controller.services.controller_interface import ControllerI
from exareme2.controller.services.flower.tasks_handler import TasksHandler
from exareme2.controller.services.strategy_interface import AlgorithmExecutionStrategyI
from exareme2.controller.uid_generator import UIDGenerator
from exareme2.worker_communication import WorkerInfo

FLOWER_SERVER_PORT = "8080"


class WorkerException(Exception):
    pass


class WorkerTaskTimeoutException(WorkerException):
    def __init__(self, timeout):
        super().__init__(
            f"Task took longer than {timeout} seconds. Increase timeout or try again."
        )


class FlowerController(ControllerI):
    def __init__(
        self, worker_landscape_aggregator, task_timeout, flower_execution_info
    ):
        super().__init__(worker_landscape_aggregator, task_timeout)
        self.flower_execution_info = flower_execution_info
        self.lock = asyncio.Lock()

    async def exec_algorithm(
        self,
        algorithm_name: str,
        algorithm_request_dto: AlgorithmRequestDTO,
        strategy: Optional[AlgorithmExecutionStrategyI] = None,
    ):
        assert strategy is None

        async with (self.lock):
            request_id = algorithm_request_dto.request_id
            context_id = UIDGenerator().get_a_uid()
            logger = ctrl_logger.get_request_logger(request_id)
            data_model = algorithm_request_dto.inputdata.data_model
            datasets = algorithm_request_dto.inputdata.datasets + (
                algorithm_request_dto.inputdata.validation_datasets
                if algorithm_request_dto.inputdata.validation_datasets
                else []
            )

            worker_ids = (
                self.worker_landscape_aggregator.get_worker_ids_with_any_of_datasets(
                    algorithm_request_dto.inputdata.data_model, datasets
                )
            )

            workers_info = [
                self.worker_landscape_aggregator.get_worker_info(worker_id)
                for worker_id in worker_ids
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

            server_task_handler.garbage_collect()
            for handler in task_handlers:
                handler.garbage_collect()

            self.flower_execution_info.set_inputdata(
                inputdata=algorithm_request_dto.inputdata.dict()
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
                    data_model,
                    datasets,
                )
                clients_pids = {
                    handler.start_flower_client(
                        algorithm_folder_path,
                        str(server_address),
                        data_model,
                        datasets,
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

    def _create_worker_tasks_handler(self, request_id, worker_info: WorkerInfo):
        worker_addr = f"{worker_info.ip}:{worker_info.port}"
        worker_db_addr = f"{worker_info.db_ip}:{worker_info.db_port}"
        return TasksHandler(
            request_id,
            worker_id=worker_info.id,
            worker_queue_addr=worker_addr,
            worker_db_addr=worker_db_addr,
            tasks_timeout=self.task_timeout,
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
