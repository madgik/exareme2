import asyncio
from typing import List

from exareme2 import flower_algorithm_folder_paths
from exareme2.controller import config as ctrl_config
from exareme2.controller.federation_info_logs import log_experiment_execution
from exareme2.controller.services.errors import WorkerTaskTimeoutError
from exareme2.controller.services.flower import FlowerController
from exareme2.controller.services.flower.tasks_handler import FlowerTasksHandler
from exareme2.controller.services.strategy_interface import AlgorithmExecutionStrategyI


class FlowerStrategy(AlgorithmExecutionStrategyI):
    _controller: FlowerController
    _local_worker_tasks_handlers: List[FlowerTasksHandler]
    _global_worker_tasks_handler: FlowerTasksHandler

    async def execute(self) -> str:
        async with self._controller.algorithm_execution_lock:
            data_model = self._algorithm_request_dto.inputdata.data_model
            datasets = self._algorithm_request_dto.inputdata.datasets + (
                self._algorithm_request_dto.inputdata.validation_datasets
                if self._algorithm_request_dto.inputdata.validation_datasets
                else []
            )

            self._global_worker_tasks_handler.garbage_collect()
            for handler in self._local_worker_tasks_handlers:
                handler.garbage_collect()

            self._controller.flower_execution_info.set_inputdata(
                inputdata=self._algorithm_request_dto.inputdata.dict()
            )
            server_pid = None
            clients_pids = {}
            server_address = f"{self._controller.worker_landscape_aggregator.get_global_worker().ip}:{ctrl_config.flower.server_port}"
            algorithm_folder_path = flower_algorithm_folder_paths[self._algorithm_name]
            try:
                server_pid = self._global_worker_tasks_handler.start_flower_server(
                    algorithm_folder_path,
                    len(self._local_worker_tasks_handlers),
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
                        ctrl_config.flower.execution_timeout,
                    ): handler
                    for handler in self._local_worker_tasks_handlers
                }

                log_experiment_execution(
                    self._logger,
                    self._request_id,
                    self._context_id,
                    self._algorithm_name,
                    self._algorithm_request_dto.inputdata.datasets,
                    self._algorithm_request_dto.parameters,
                    [h.worker_id for h in self._local_worker_tasks_handlers],
                )
                result = (
                    await self._controller.flower_execution_info.get_result_with_timeout()
                )

                self._logger.info(
                    f"Finished execution -> {self._algorithm_name} with {self._request_id}"
                )

                return result

            except asyncio.TimeoutError:
                raise WorkerTaskTimeoutError()
            finally:
                await self._cleanup(
                    self._algorithm_name,
                    self._global_worker_tasks_handler,
                    server_pid,
                    clients_pids,
                )

    async def _cleanup(
        self, algorithm_name, server_task_handler, server_pid, clients_pids
    ):
        await self._controller.flower_execution_info.reset()
        server_task_handler.stop_flower_server(server_pid, algorithm_name)
        for pid, handler in clients_pids.items():
            handler.stop_flower_client(pid, algorithm_name)
