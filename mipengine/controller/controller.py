import asyncio
import concurrent.futures
import logging

import datetime
import random

from typing import List

from pydantic import BaseModel

from mipengine.controller.node_tasks_handler_celery import NodeTasksHandlerCelery
from mipengine.controller.algorithm_executor import AlgorithmExecutor
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO
from mipengine.controller.algorithm_execution_DTOs import (
    AlgorithmExecutionDTO,
    NodesTasksHandlersDTO,
)
from mipengine.controller.node_registry import node_registry
from mipengine.controller import config as controller_config
from mipengine.controller.api.validator import validate_algorithm_request
from mipengine.controller import controller_logger as ctrl_logger


CLEANUP_INTERVAL = 10


class Controller:
    def __init__(self):
        self._nodes_for_cleanup = {}
        self._keep_cleaning_up = True
        self._loggers = {}

    async def exec_algorithm(
        self, algorithm_name: str, algorithm_request_dto: AlgorithmRequestDTO
    ) -> str:

        context_id = get_a_uniqueid()

        logger = ctrl_logger.get_request_logger(context_id=context_id)
        self._loggers[context_id] = logger

        node_tasks_handlers = self._get_nodes_tasks_handlers(
            pathology=algorithm_request_dto.inputdata.pathology,
            datasets=algorithm_request_dto.inputdata.datasets,
        )

        algo_execution_node_ids = [
            node_tasks_handlers.global_node_tasks_handler.node_id
        ]
        for local_node_task_handler in node_tasks_handlers.local_nodes_tasks_handlers:
            algo_execution_node_ids.append(local_node_task_handler.node_id)

        try:
            algorithm_result = await self._exec_algorithm_with_task_handlers(
                context_id=context_id,
                algorithm_name=algorithm_name,
                algorithm_request_dto=algorithm_request_dto,
                tasks_handlers=node_tasks_handlers,
                logger=logger,
            )
        except Exception as exc:
            raise exc

        finally:
            self._append_context_id_for_cleanup(
                context_id=context_id, node_ids=algo_execution_node_ids
            )

        return algorithm_result

    def _append_context_id_for_cleanup(self, context_id: str, node_ids: List[str]):
        if context_id not in self._nodes_for_cleanup.keys():
            self._nodes_for_cleanup[context_id] = node_ids
        else:
            # getting in here would mean that an algorithm with the same context_id has
            # finished and is currently in the cleanup process, this indicates context_id
            # collision.
            self._loggers[context_id].warning(
                f"An algorithm with the same {context_id=} was previously executed and"
                f"it is still in the cleanup process. This should not happen..."
            )
            for node_id in node_ids:
                self._nodes_for_cleanup[context_id].append(node_id)

    async def start_cleanup_loop(self):
        print("starting cleanup_loop")  # TODO what logger whould be used here?
        self._keep_cleaning_up = True
        task = asyncio.create_task(self.cleanup_loop())
        print("started clean_up loop")
        return task

    async def stop_cleanup_loop(self):
        self._keep_cleaning_up = False

    async def cleanup_loop(self):
        while self._keep_cleaning_up:
            cleaned_up_nodes = {}
            for context_id, node_ids in self._nodes_for_cleanup.items():
                cleaned_up_nodes[context_id] = []
                for node_id in node_ids:
                    try:
                        node_info = _get_node_info_by_id(node_id)
                        task_handler = _create_node_task_handler(node_info)
                        task_handler.clean_up(context_id)

                        self._loggers[context_id].debug(
                            f"clean_up task succeeded for {node_id=}  for {context_id=}"
                        )
                        cleaned_up_nodes[context_id].append(node_id)
                    except Exception as exc:
                        self._loggers[context_id].debug(
                            f"clean_up task FAILED for {node_id=} "
                            f"for {context_id=}. Will retry in a while... fail "
                            f"reason: {type(exc)}:{exc}"
                        )

            for context_id, node_ids in cleaned_up_nodes.items():
                for node_id in node_ids:
                    self._nodes_for_cleanup[context_id].remove(node_id)
                if not self._nodes_for_cleanup[context_id]:
                    self._nodes_for_cleanup.pop(context_id)

            await asyncio.sleep(CLEANUP_INTERVAL)

    async def _exec_algorithm_with_task_handlers(
        self,
        context_id: str,
        algorithm_name: str,
        algorithm_request_dto: AlgorithmRequestDTO,
        tasks_handlers: NodesTasksHandlersDTO,
        logger: logging.Logger,  # TODO logger should not be passed here
    ) -> str:

        # TODO: AlgorithmExecutor is not yet implemented with asyncio. This is a
        # temprorary solution for not blocking the calling function
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
        available_datasets_per_schema = self.get_all_available_datasets_per_schema()
        validate_algorithm_request(
            algorithm_name=algorithm_name,
            algorithm_request_dto=algorithm_request_dto,
            available_datasets_per_schema=available_datasets_per_schema,
        )

    async def start_node_registry(self):
        print("starting node registry")  # TODO what logger should be used here?
        node_registry.keep_updating = True
        asyncio.create_task(node_registry.update())
        print("started node registry")

    async def stop_node_registry(self):
        node_registry.keep_updating = False

    def get_all_datasets_per_node(self):
        datasets = {}
        for node in node_registry.get_all_local_nodes():
            datasets[node.id] = node.datasets_per_schema
        return datasets

    def get_all_available_schemas(self):
        return node_registry.get_all_available_schemas()

    def get_all_available_datasets_per_schema(self):
        return node_registry.get_all_available_datasets_per_schema()

    def get_all_local_nodes(self):
        return node_registry.get_all_local_nodes()

    def _get_nodes_tasks_handlers(
        self, pathology: str, datasets: List[str]
    ) -> NodesTasksHandlersDTO:

        global_node = node_registry.get_all_global_nodes()[0]
        global_node_tasks_handler = _create_node_task_handler(
            _NodeInfoDTO(
                node_id=global_node.id,
                queue_address=":".join([str(global_node.ip), str(global_node.port)]),
                db_address=":".join([str(global_node.db_ip), str(global_node.db_port)]),
                tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
            )
        )

        # Get only the relevant nodes from the node registry
        local_nodes_info = _get_nodes_info_by_dataset(
            pathology=pathology, datasets=datasets
        )
        local_nodes_tasks_handlers = [
            _create_node_task_handler(task_handler) for task_handler in local_nodes_info
        ]

        return NodesTasksHandlersDTO(
            global_node_tasks_handler=global_node_tasks_handler,
            local_nodes_tasks_handlers=local_nodes_tasks_handlers,
        )


class _NodeInfoDTO(BaseModel):
    node_id: str
    queue_address: str
    db_address: str
    tasks_timeout: int

    class Config:
        allow_mutation = False


def _get_node_info_by_id(node_id: str) -> _NodeInfoDTO:
    global_nodes = node_registry.get_all_global_nodes()
    local_nodes = node_registry.get_all_local_nodes()

    for node in global_nodes + local_nodes:
        if node.id == node_id:
            return _NodeInfoDTO(
                node_id=node.id,
                queue_address=":".join([str(node.ip), str(node.port)]),
                db_address=":".join([str(node.db_ip), str(node.db_port)]),
                tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
            )


def _get_nodes_info_by_dataset(
    pathology: str, datasets: List[str]
) -> List[_NodeInfoDTO]:
    local_nodes = node_registry.get_nodes_with_any_of_datasets(
        schema=pathology,
        datasets=datasets,
    )
    nodes_info = []
    for local_node in local_nodes:
        nodes_info.append(
            _NodeInfoDTO(
                node_id=local_node.id,
                queue_address=":".join([str(local_node.ip), str(local_node.port)]),
                db_address=":".join([str(local_node.db_ip), str(local_node.db_port)]),
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
    uid = datetime.datetime.now().microsecond + (random.randrange(1, 100 + 1) * 100000)
    return f"{uid}"
