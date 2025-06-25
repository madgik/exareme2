import asyncio
from typing import List

from exareme2 import flower_algorithm_folder_paths
from exareme2.controller import config as ctrl_config
from exareme2.controller.federation_info_logs import log_experiment_execution
from exareme2.controller.services.flower import FlowerController
from exareme2.controller.services.flower.tasks_handler import FlowerTasksHandler
from exareme2.controller.services.strategy_interface import AlgorithmExecutionStrategyI

# TODO Kostas, move to controller config
FLOWER_SERVER_PORT = "8080"


class WorkerTaskTimeoutException(Exception):
    def __init__(self, timeout):
        super().__init__(
            f"Task took longer than {timeout} seconds. Increase timeout or try again."
        )


class FlowerStrategy(AlgorithmExecutionStrategyI):
    controller: FlowerController
    local_worker_tasks_handlers: List[FlowerTasksHandler]
    global_worker_tasks_handler: FlowerTasksHandler

    async def execute(self) -> str:
        async with (self.controller.algorithm_execution_lock):
            data_model = self.algorithm_request_dto.inputdata.data_model
            datasets = self.algorithm_request_dto.inputdata.datasets + (
                self.algorithm_request_dto.inputdata.validation_datasets
                if self.algorithm_request_dto.inputdata.validation_datasets
                else []
            )

            self.global_worker_tasks_handler.garbage_collect()
            for handler in self.local_worker_tasks_handlers:
                handler.garbage_collect()

            self.controller.flower_execution_info.set_inputdata(
                inputdata=self.algorithm_request_dto.inputdata.dict()
            )
            server_pid = None
            clients_pids = {}
            server_address = f"{self.controller.worker_landscape_aggregator.get_global_worker().ip}:{FLOWER_SERVER_PORT}"
            algorithm_folder_path = flower_algorithm_folder_paths[self.algorithm_name]
            try:
                server_pid = self.global_worker_tasks_handler.start_flower_server(
                    algorithm_folder_path,
                    len(self.local_worker_tasks_handlers),
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
                    for handler in self.local_worker_tasks_handlers
                }

                log_experiment_execution(
                    self.logger,
                    self.request_id,
                    self.context_id,
                    self.algorithm_name,
                    self.algorithm_request_dto.inputdata.datasets,
                    self.algorithm_request_dto.parameters,
                    [h.worker_id for h in self.local_worker_tasks_handlers],
                )
                result = (
                    await self.controller.flower_execution_info.get_result_with_timeout()
                )

                self.logger.info(
                    f"Finished execution -> {self.algorithm_name} with {self.request_id}"
                )

                # TODO Kostas, The result should be str but this returns dict
                return result

            except asyncio.TimeoutError:
                raise WorkerTaskTimeoutException(self.controller.task_timeout)
            finally:
                await self._cleanup(
                    self.algorithm_name,
                    self.global_worker_tasks_handler,
                    server_pid,
                    clients_pids,
                )

    async def _cleanup(
        self, algorithm_name, server_task_handler, server_pid, clients_pids
    ):
        await self.controller.flower_execution_info.reset()
        server_task_handler.stop_flower_server(server_pid, algorithm_name)
        for pid, handler in clients_pids.items():
            handler.stop_flower_client(pid, algorithm_name)
