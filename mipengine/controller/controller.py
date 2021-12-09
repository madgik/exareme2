import asyncio
import concurrent.futures

import datetime
import random

from typing import Dict, List, Tuple, Optional, Any

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


class Controller:
    def __init__(self):
        # TODO start node registry here?
        pass

    async def exec_algorithm(
        self, algorithm_name: str, algorithm_request_dto: AlgorithmRequestDTO
    ):
        context_id = get_a_uniqueid()
        logger=ctrl_logger.get_request_logger(context_id=context_id)

        all_nodes_tasks_handlers = self._create_nodes_tasks_handlers(
            context_id=context_id,
            pathology=algorithm_request_dto.inputdata.pathology,
            datasets=algorithm_request_dto.inputdata.datasets,
        )

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
            all_nodes_tasks_handlers,
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
        asyncio.create_task(node_registry.update())

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

    def _create_nodes_tasks_handlers(
        self, context_id: str, pathology: str, datasets: List[str]
    ) -> NodesTasksHandlersDTO:

        # Get only the relevant nodes from the node registry
        global_node = node_registry.get_all_global_nodes()[0]
        local_nodes = node_registry.get_nodes_with_any_of_datasets(
            schema=pathology,
            datasets=datasets,
        )

        queue_address = ":".join([str(global_node.ip), str(global_node.port)])
        db_address = ":".join([str(global_node.db_ip), str(global_node.db_port)])
        tasks_timeout = controller_config.rabbitmq.celery_tasks_timeout
        # Instantiate the INodeTasksHandler for the Global Node
        global_node_tasks_handler = NodeTasksHandlerCelery(
            node_id=global_node.id,
            node_queue_addr=queue_address,
            node_db_addr=db_address,
            tasks_timeout=tasks_timeout,
        )

        local_nodes_tasks_handlers = []
        for local_node in local_nodes:
            # Instantiate the INodeTasksHandlers for the Local Nodes
            queue_address = ":".join([str(local_node.ip), str(local_node.port)])
            db_address = ":".join([str(local_node.db_ip), str(local_node.db_port)])
            tasks_timeout = controller_config.rabbitmq.celery_tasks_timeout

            node_tasks_handler = NodeTasksHandlerCelery(
                node_id=local_node.id,
                node_queue_addr=queue_address,
                node_db_addr=db_address,
                tasks_timeout=tasks_timeout,
            )

            local_nodes_tasks_handlers.append(node_tasks_handler)

        return NodesTasksHandlersDTO(
            global_node_tasks_handler=global_node_tasks_handler,
            local_nodes_tasks_handlers=local_nodes_tasks_handlers,
        )


def get_a_uniqueid():
    return "{}".format(
        datetime.datetime.now().microsecond + (random.randrange(1, 100 + 1) * 100000)
    )
