import asyncio
import concurrent.futures
import logging
import random
from datetime import datetime
from typing import List

from pydantic import BaseModel

from mipengine.controller import config as controller_config
from mipengine.controller import controller_logger as ctrl_logger
from mipengine.controller.algorithm_execution_DTOs import AlgorithmExecutionDTO
from mipengine.controller.algorithm_execution_DTOs import NodesTasksHandlersDTO
from mipengine.controller.algorithm_executor import AlgorithmExecutor
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO
from mipengine.controller.api.validator import validate_algorithm_request
from mipengine.controller.cleaner import Cleaner
from mipengine.controller.node_registry import node_registry
from mipengine.controller.node_tasks_handler_celery import NodeTasksHandlerCelery


class _NodeInfoDTO(BaseModel):
    node_id: str
    queue_address: str
    db_address: str
    tasks_timeout: int

    class Config:
        allow_mutation = False


class Controller:
    def __init__(self):
        self._node_registry = node_registry
        self._controller_logger = ctrl_logger.get_background_service_logger()
        self._cleaner = Cleaner(node_registry=self._node_registry)

    async def start_cleanup_loop(self):
        self._controller_logger.info("starting cleanup_loop")
        self._cleaner.keep_cleaning_up = True
        task = asyncio.create_task(self._cleaner.cleanup_loop())
        self._controller_logger.info("started clean_up loop")
        return task

    async def stop_cleanup_loop(self):
        self._cleaner.keep_cleaning_up = False

    async def exec_algorithm(
        self,
        request_id: str,
        algorithm_name: str,
        algorithm_request_dto: AlgorithmRequestDTO,
    ):
        context_id = get_a_uniqueid()
        algo_execution_logger = ctrl_logger.get_request_logger(request_id=request_id)

        data_model = algorithm_request_dto.inputdata.data_model
        datasets = algorithm_request_dto.inputdata.datasets

        node_tasks_handlers = self._get_nodes_tasks_handlers(
            data_model=data_model, datasets=datasets
        )

        algo_execution_node_ids = [
            node_tasks_handlers.global_node_tasks_handler.node_id
        ]
        for local_node_task_handler in node_tasks_handlers.local_nodes_tasks_handlers:
            algo_execution_node_ids.append(local_node_task_handler.node_id)

        self._cleaner._add_contextid_for_cleanup(context_id, algo_execution_node_ids)

        try:
            algorithm_result = await self._exec_algorithm_with_task_handlers(
                request_id=request_id,
                context_id=context_id,
                algorithm_name=algorithm_name,
                algorithm_request_dto=algorithm_request_dto,
                tasks_handlers=node_tasks_handlers,
                logger=algo_execution_logger,
            )
        finally:
            self._cleaner._release_contextid_for_cleanup(context_id=context_id)

        return algorithm_result

    async def _exec_algorithm_with_task_handlers(
        self,
        request_id: str,
        context_id: str,
        algorithm_name: str,
        algorithm_request_dto: AlgorithmRequestDTO,
        tasks_handlers: NodesTasksHandlersDTO,
        logger: logging.Logger,
    ) -> str:

        # TODO: AlgorithmExecutor is not yet implemented with asyncio. This is a
        # temporary solution for not blocking the calling function
        def run_algorithm_executor_in_threadpool(
            algorithm_execution_dto: AlgorithmExecutionDTO,
            all_nodes_tasks_handlers: NodesTasksHandlersDTO,
        ):
            algorithm_executor = AlgorithmExecutor(
                algorithm_execution_dto, all_nodes_tasks_handlers
            )

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(algorithm_executor.run)
                result = future.result()
                return result

        algorithm_execution_dto = AlgorithmExecutionDTO(
            request_id=request_id,
            context_id=context_id,
            algorithm_name=algorithm_name,
            algorithm_request_dto=algorithm_request_dto,
        )

        loop = asyncio.get_running_loop()

        logger.info(f"starts executing->  {algorithm_name=}")

        algorithm_result = await loop.run_in_executor(
            None,
            run_algorithm_executor_in_threadpool,
            algorithm_execution_dto,
            tasks_handlers,
        )

        logger.info(f"finished execution->  {algorithm_name=}")
        logger.info(f"algorithm result-> {algorithm_result.json()=}")

        return algorithm_result.json()

    def validate_algorithm_execution_request(
        self, algorithm_name: str, algorithm_request_dto: AlgorithmRequestDTO
    ):
        available_datasets_per_data_model = (
            self.get_all_available_datasets_per_data_model()
        )
        validate_algorithm_request(
            algorithm_name=algorithm_name,
            algorithm_request_dto=algorithm_request_dto,
            available_datasets_per_data_model=available_datasets_per_data_model,
        )

    async def start_node_registry(self):
        self._controller_logger.info("starting node registry")
        self._node_registry.keep_updating = True
        asyncio.create_task(self._node_registry.update())
        self._controller_logger.info("started node registry")

    async def stop_node_registry(self):
        self._node_registry.keep_updating = False

    def get_all_datasets_per_node(self):
        datasets = {}
        for node in self._node_registry.get_all_local_nodes():
            datasets[node.id] = node.datasets_per_data_model
        return datasets

    def get_all_available_schemas(self):
        return self._node_registry.get_all_available_data_models()

    def get_all_available_datasets_per_data_model(self):
        return self._node_registry.get_all_available_datasets_per_data_model()

    def get_all_local_nodes(self):
        return self._node_registry.get_all_local_nodes()

    def get_global_node(self):
        global_nodes = self._node_registry.get_all_global_nodes()
        if global_nodes:
            return global_nodes[0]

    def _get_nodes_tasks_handlers(
        self, data_model: str, datasets: List[str]
    ) -> NodesTasksHandlersDTO:
        global_node = self._node_registry.get_all_global_nodes()[0]
        global_node_tasks_handler = _create_node_task_handler(
            _NodeInfoDTO(
                node_id=global_node.id,
                queue_address=":".join([str(global_node.ip), str(global_node.port)]),
                db_address=":".join([str(global_node.db_ip), str(global_node.db_port)]),
                tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
            )
        )

        # Get only the relevant nodes from the node registry
        local_nodes_info = self._get_nodes_info_by_dataset(
            data_model=data_model, datasets=datasets
        )
        local_nodes_tasks_handlers = [
            _create_node_task_handler(task_handler) for task_handler in local_nodes_info
        ]

        return NodesTasksHandlersDTO(
            global_node_tasks_handler=global_node_tasks_handler,
            local_nodes_tasks_handlers=local_nodes_tasks_handlers,
        )

    def _get_nodes_info_by_dataset(
        self, data_model: str, datasets: List[str]
    ) -> List[_NodeInfoDTO]:
        local_nodes = self._node_registry.get_nodes_with_any_of_datasets(
            data_model=data_model,
            datasets=datasets,
        )
        nodes_info = []
        for local_node in local_nodes:
            nodes_info.append(
                _NodeInfoDTO(
                    node_id=local_node.id,
                    queue_address=":".join([str(local_node.ip), str(local_node.port)]),
                    db_address=":".join(
                        [str(local_node.db_ip), str(local_node.db_port)]
                    ),
                    tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
                )
            )

        return nodes_info


def _create_node_task_handler(node_info: _NodeInfoDTO) -> NodeTasksHandlerCelery:
    return NodeTasksHandlerCelery(
        node_id=node_info.node_id,
        node_queue_addr=node_info.queue_address,
        node_db_addr=node_info.db_address,
        tasks_timeout=node_info.tasks_timeout,
    )


def get_a_uniqueid() -> str:
    uid = datetime.now().microsecond + (random.randrange(1, 100 + 1) * 100000)
    return f"{uid}"
