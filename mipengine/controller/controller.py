import asyncio
import concurrent.futures

import datetime
import random

import logging
import traceback

from typing import Dict, List, Tuple, Optional, Any

from mipengine.controller.node_tasks_handler_celery import (
    NodeTasksHandlerCelery,
    CeleryParamsDTO,
)
from mipengine.controller.algorithm_executor import AlgorithmExecutor

from mipengine.controller.algorithm_execution_DTOs import (
    AlgorithmRequestDTO,
    AlgorithmExecutionDTO,
    NodesTasksHandlersDTO,
)

from mipengine.controller.node_registry import node_registry
from mipengine.controller import config as controller_config


class Controller:
    def __init__(self):
        # TODO start node registry here?
        pass

    async def exec_algorithm(
        self, algorithm_name: str, algorithm_request_dto: AlgorithmRequestDTO
    ):

        context_id = get_a_uniqueid()

        all_nodes_tasks_handlers = self._create_nodes_tasks_handlers(
            context_id=context_id,
            pathology=algorithm_request_dto.pathology,
            datasets=algorithm_request_dto.datasets,
        )

        # TODO: AlgorithmExecutor is not yest implemented with asyncio. This is a
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

        try:
            algorithm_execution_dto = AlgorithmExecutionDTO(
                context_id=context_id,
                algorithm_name=algorithm_name,
                algorithm_request_dto=algorithm_request_dto,
            )

            loop = asyncio.get_running_loop()

            #DEBUG(future logging..)
            print(
                f"\n(controller.py::exec_algorithm) starting executing->  {algorithm_name=} with {context_id=}\n"
            )
            #DEBUG end
            
            algorithm_result = await loop.run_in_executor(
                None,
                run_algorithm_executor_in_threadpool,
                algorithm_execution_dto,
                all_nodes_tasks_handlers,
            )

            #DEBUG(future logging..)
            print(
                f"\n(controller.py::exec_algorithm) FINISHED->  {algorithm_name=} with {context_id=}"
            )
            print(f"{algorithm_result.json()=}\n")
            #DEBUG end
            
            return algorithm_result.json()

        except:
            logging.error(
                f"Algorithm execution failed. Unexpected exception: \n {traceback.format_exc()}"
            )

    def _create_nodes_tasks_handlers(
        self, context_id: str, pathology: str, datasets: List[str]
    ) -> NodesTasksHandlersDTO:

        # Get only teh relevant nodes from teh node registry
        global_node = node_registry.get_all_global_nodes()[0]
        local_nodes = node_registry.get_nodes_with_any_of_datasets(
            schema=pathology,
            datasets=datasets,
        )

        # Global Node, gather the Celery Parameters
        global_node_celery_params_dto = CeleryParamsDTO(
            task_queue_domain=global_node.ip,
            task_queue_port=global_node.port,
            db_domain=global_node.db_ip,
            db_port=global_node.db_port,
            user=controller_config.rabbitmq.user,
            password=controller_config.rabbitmq.password,
            vhost=controller_config.rabbitmq.vhost,
            max_retries=controller_config.rabbitmq.celery_tasks_max_retries,
            interval_start=controller_config.rabbitmq.celery_tasks_interval_start,
            interval_step=controller_config.rabbitmq.celery_tasks_interval_step,
            interval_max=controller_config.rabbitmq.celery_tasks_interval_max,
        )
        # Instantiate the INodeTasksHandler for the Global Node
        global_node_tasks_handler = NodeTasksHandlerCelery(
            node_id=global_node.id, celery_params=global_node_celery_params_dto
        )

        local_nodes_tasks_handlers = []
        for local_node in local_nodes:
            # Local Nodes, gather the Celery Parameters
            celery_params = CeleryParamsDTO(
                task_queue_domain=local_node.ip,
                task_queue_port=local_node.port,
                db_domain=local_node.db_ip,
                db_port=local_node.db_port,
                user=controller_config.rabbitmq.user,
                password=controller_config.rabbitmq.password,
                vhost=controller_config.rabbitmq.vhost,
                max_retries=controller_config.rabbitmq.celery_tasks_max_retries,
                interval_start=controller_config.rabbitmq.celery_tasks_interval_start,
                interval_step=controller_config.rabbitmq.celery_tasks_interval_step,
                interval_max=controller_config.rabbitmq.celery_tasks_interval_max,
            )

            # Instantiate the INodeTasksHandlers for the Local Nodes
            node_tasks_handler = NodeTasksHandlerCelery(
                node_id=local_node.id, celery_params=celery_params
            )
            local_nodes_tasks_handlers.append(node_tasks_handler)

        return NodesTasksHandlersDTO(
            global_node_tasks_handler=global_node_tasks_handler,
            local_nodes_tasks_handlers=local_nodes_tasks_handlers,
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

def get_a_uniqueid():
    return "{}".format(
        datetime.datetime.now().microsecond + (random.randrange(1, 100 + 1) * 100000)
    )
