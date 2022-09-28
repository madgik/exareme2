import asyncio
import concurrent
import logging
from datetime import datetime
import random
import uuid
import itertools
from typing import Dict
from typing import List

from pydantic import BaseModel

from mipengine.controller import config as controller_config
from mipengine.controller import controller_logger as ctrl_logger
from mipengine.controller.algorithm_execution_DTOs import AlgorithmExecutionDTO
from mipengine.controller.algorithm_execution_DTOs import NodesTasksHandlersDTO
from mipengine.controller.algorithm_execution_tasks_handler import (
    NodeAlgorithmTasksHandler,
)
from mipengine.controller.algorithm_executor import AlgorithmExecutor
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO
from mipengine.controller.api.validator import validate_algorithm_request
from mipengine.controller.cleaner import Cleaner
from mipengine.controller.federation_info_logs import log_experiment_execution
from mipengine.controller.node_landscape_aggregator import DataModelsCDES
from mipengine.controller.node_landscape_aggregator import DatasetsLocations
from mipengine.controller.node_landscape_aggregator import NodeLandscapeAggregator
from mipengine.node_info_DTOs import NodeInfo


class _NodeInfoDTO(BaseModel):
    node_id: str
    queue_address: str
    db_address: str
    tasks_timeout: int
    run_udf_task_timeout: int

    class Config:
        allow_mutation = False


class Controller:
    def __init__(self):
        self._controller_logger = ctrl_logger.get_background_service_logger()

        self._node_landscape_aggregator = NodeLandscapeAggregator()
        self._cleaner = Cleaner()

        self._executor = concurrent.futures.ThreadPoolExecutor()

    def start_cleanup_loop(self):
        self._controller_logger.info("(Controller) Cleaner starting ...")
        self._cleaner.start()
        self._controller_logger.info("(Controller) Cleaner started.")

    def stop_cleanup_loop(self):
        self._cleaner.stop()

    def start_node_landscape_aggregator(self):
        self._controller_logger.info(
            "(Controller) NodeLandscapeAggregator starting ..."
        )
        self._node_landscape_aggregator.start()
        self._controller_logger.info("(Controller) NodeLandscapeAggregator started.")

    def stop_node_landscape_aggregator(self):
        self._node_landscape_aggregator.stop()

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

        algo_execution_local_node_ids = [
            local_node_task_handler.node_id
            for local_node_task_handler in node_tasks_handlers.local_nodes_tasks_handlers
        ]

        algo_execution_node_ids = algo_execution_local_node_ids
        if node_tasks_handlers.global_node_tasks_handler:
            algo_execution_node_ids.append(
                node_tasks_handlers.global_node_tasks_handler.node_id
            )

        self._cleaner.add_contextid_for_cleanup(context_id, algo_execution_node_ids)

        datasets_per_local_node: Dict[str, List[str]] = {
            task_handler.node_id: self._node_landscape_aggregator.get_node_specific_datasets(
                task_handler.node_id, data_model, datasets
            )
            for task_handler in node_tasks_handlers.local_nodes_tasks_handlers
        }

        log_experiment_execution(
            logger=algo_execution_logger,
            request_id=request_id,
            context_id=context_id,
            algorithm_name=algorithm_name,
            datasets=algorithm_request_dto.inputdata.datasets,
            algorithm_parameters=algorithm_request_dto.json(),
            local_node_ids=algo_execution_local_node_ids,
        )

        try:
            algorithm_result = await self._exec_algorithm_with_task_handlers(
                request_id=request_id,
                context_id=context_id,
                algorithm_name=algorithm_name,
                algorithm_request_dto=algorithm_request_dto,
                tasks_handlers=node_tasks_handlers,
                datasets_per_local_node=datasets_per_local_node,
                logger=algo_execution_logger,
            )
        finally:
            self._cleaner.release_context_id(context_id=context_id)

        return algorithm_result

    async def _exec_algorithm_with_task_handlers(
        self,
        request_id: str,
        context_id: str,
        algorithm_name: str,
        algorithm_request_dto: AlgorithmRequestDTO,
        tasks_handlers: NodesTasksHandlersDTO,
        datasets_per_local_node: Dict[str, List[str]],
        logger: logging.Logger,
    ) -> str:
        algorithm_execution_dto = AlgorithmExecutionDTO(
            request_id=request_id,
            context_id=context_id,
            algorithm_name=algorithm_name,
            data_model=algorithm_request_dto.inputdata.data_model,
            datasets_per_local_node=datasets_per_local_node,
            x_vars=algorithm_request_dto.inputdata.x,
            y_vars=algorithm_request_dto.inputdata.y,
            var_filters=algorithm_request_dto.inputdata.filters,
            algo_parameters=algorithm_request_dto.parameters,
            algo_flags=algorithm_request_dto.flags,
        )
        algorithm_executor = AlgorithmExecutor(
            algorithm_execution_dto,
            tasks_handlers,
            self._node_landscape_aggregator.get_cdes(
                algorithm_execution_dto.data_model
            ),
        )

        # By calling blocking method AlgorithmExecutor.run() inside run_in_executor(),
        # AlgorithmExecutor.run() will run in a separate thread of the
        # threadpool and at the same time yield control to the executor event loop, through await
        loop = asyncio.get_event_loop()
        algorithm_result = await loop.run_in_executor(
            self._executor, algorithm_executor.run
        )

        logger.info(f"Finished execution->  {algorithm_name=} with {request_id=}")
        logger.debug(f"Algorithm {request_id=} result-> {algorithm_result.json()=}")

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

    def get_datasets_locations(self) -> DatasetsLocations:
        return self._node_landscape_aggregator.get_datasets_locations()

    def get_cdes_per_data_model(self) -> dict:
        return {
            data_model: {
                column: metadata.dict() for column, metadata in cdes.values.items()
            }
            for data_model, cdes in self._node_landscape_aggregator.get_cdes_per_data_model().data_models_cdes.items()
        }

    def get_all_available_data_models(self) -> List[str]:
        return list(
            self._node_landscape_aggregator.get_cdes_per_data_model().data_models_cdes.keys()
        )

    def get_all_available_datasets_per_data_model(self) -> Dict[str, List[str]]:
        return (
            self._node_landscape_aggregator.get_all_available_datasets_per_data_model()
        )

    def get_all_local_nodes(self) -> List[NodeInfo]:
        return self._node_landscape_aggregator.get_all_local_nodes()

    def get_global_node(self) -> NodeInfo:
        return self._node_landscape_aggregator.get_global_node()

    def _get_nodes_tasks_handlers(
        self, data_model: str, datasets: List[str]
    ) -> NodesTasksHandlersDTO:
        # Get only the relevant nodes
        local_nodes_info = self._get_nodes_info_by_dataset(
            data_model=data_model, datasets=datasets
        )
        local_nodes_tasks_handlers = [
            _create_node_task_handler(node_info) for node_info in local_nodes_info
        ]

        # If there is only one localnode that is going to be used, there is no need for a global node.
        global_node_tasks_handler = None
        if len(local_nodes_tasks_handlers) > 1:
            global_node = self._node_landscape_aggregator.get_global_node()
            global_node_tasks_handler = _create_node_task_handler(
                _NodeInfoDTO(
                    node_id=global_node.id,
                    queue_address=":".join(
                        [str(global_node.ip), str(global_node.port)]
                    ),
                    db_address=":".join(
                        [str(global_node.db_ip), str(global_node.db_port)]
                    ),
                    tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
                    run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
                )
            )

        return NodesTasksHandlersDTO(
            global_node_tasks_handler=global_node_tasks_handler,
            local_nodes_tasks_handlers=local_nodes_tasks_handlers,
        )

    def _get_node_info_by_id(self, node_id: str) -> _NodeInfoDTO:
        node = self._node_landscape_aggregator.get_node_info(node_id)
        return _NodeInfoDTO(
            node_id=node.id,
            queue_address=":".join([str(node.ip), str(node.port)]),
            db_address=":".join([str(node.db_ip), str(node.db_port)]),
            tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
            run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
        )

    def _get_nodes_info_by_dataset(
        self, data_model: str, datasets: List[str]
    ) -> List[_NodeInfoDTO]:
        local_node_ids = (
            self._node_landscape_aggregator.get_node_ids_with_any_of_datasets(
                data_model=data_model,
                datasets=datasets,
            )
        )
        local_nodes_info = [
            self._node_landscape_aggregator.get_node_info(node_id)
            for node_id in local_node_ids
        ]
        nodes_info = []
        for local_node in local_nodes_info:
            nodes_info.append(
                _NodeInfoDTO(
                    node_id=local_node.id,
                    queue_address=":".join([str(local_node.ip), str(local_node.port)]),
                    db_address=":".join(
                        [str(local_node.db_ip), str(local_node.db_port)]
                    ),
                    tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
                    run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
                )
            )

        return nodes_info


def _create_node_task_handler(node_info: _NodeInfoDTO) -> NodeAlgorithmTasksHandler:
    return NodeAlgorithmTasksHandler(
        node_id=node_info.node_id,
        node_queue_addr=node_info.queue_address,
        node_db_addr=node_info.db_address,
        tasks_timeout=node_info.tasks_timeout,
        run_udf_task_timeout=node_info.run_udf_task_timeout,
    )


def _get_a_uniqueid() -> str:
    numchars = 10
    maxid = 10**numchars
    yield from itertools.cycle(str(i).rjust(numchars, "0") for i in range(maxid))


uniqueid_gen = _get_a_uniqueid()


def get_a_uniqueid():
    return next(uniqueid_gen)
