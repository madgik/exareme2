import asyncio
import concurrent
import logging
import traceback
from typing import Dict
from typing import List
from typing import Tuple

from pydantic import BaseModel

from mipengine import algorithm_classes
from mipengine.algorithms.algorithm import Algorithm
from mipengine.algorithms.algorithm import AlgorithmDTO
from mipengine.algorithms.algorithm import Variables
from mipengine.controller import config as controller_config
from mipengine.controller import controller_logger as ctrl_logger
from mipengine.controller.algorithm_execution_DTOs import NodesTasksHandlersDTO
from mipengine.controller.algorithm_execution_tasks_handler import (
    NodeAlgorithmTasksHandler,
)
from mipengine.controller.algorithm_executor import AlgorithmExecutor
from mipengine.controller.algorithm_executor import AlgorithmExecutorDTO
from mipengine.controller.algorithm_executor import AlgorithmExecutorSingleLocalNode
from mipengine.controller.algorithm_executor_nodes import GlobalNode
from mipengine.controller.algorithm_executor_nodes import LocalNode
from mipengine.controller.algorithm_flow_data_objects import LocalNodesData
from mipengine.controller.algorithm_flow_data_objects import LocalNodesTable
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO
from mipengine.controller.api.validator import validate_algorithm_request
from mipengine.controller.celery_app import CeleryConnectionError
from mipengine.controller.celery_app import CeleryTaskTimeoutException
from mipengine.controller.cleaner import Cleaner
from mipengine.controller.federation_info_logs import log_experiment_execution
from mipengine.controller.node_landscape_aggregator import DatasetsLocations
from mipengine.controller.node_landscape_aggregator import NodeLandscapeAggregator
from mipengine.controller.uid_generator import UIDGenerator
from mipengine.exceptions import InsufficientDataError
from mipengine.node_info_DTOs import NodeInfo
from mipengine.node_tasks_DTOs import NodeSMPCDTO
from mipengine.node_tasks_DTOs import NodeTableDTO
from mipengine.node_tasks_DTOs import NodeUDFDTO
from mipengine.node_tasks_DTOs import TableInfo

# node_landscape_aggregator = NodeLandscapeAggregator()


class Nodes:
    def __init__(self, global_node, local_nodes):
        self._global_node = global_node
        self._local_nodes = local_nodes

    @property
    def global_node(self):
        return self._global_node

    @property
    def local_nodes(self):
        return self._local_nodes


class _NodeInfoDTO(BaseModel):
    node_id: str
    queue_address: str
    db_address: str
    tasks_timeout: int
    run_udf_task_timeout: int

    class Config:
        allow_mutation = False


class CommandIdGenerator:
    def __init__(self):
        self._index = 0

    def get_next_command_id(self) -> int:
        current = self._index
        self._index += 1
        return current


class RequestConstraintsError(Exception):
    def __init__(self, algorithm_request_dto):
        message = f"None of the nodes has enogh data to execute request: {algorithm_request_dto} "
        super().__init__(message)
        self.message = message


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
        # node_landscape_aggregator.start()
        self._node_landscape_aggregator.start()
        self._controller_logger.info("(Controller) NodeLandscapeAggregator started.")

    def stop_node_landscape_aggregator(self):
        # node_landscape_aggregator.stop()
        self._node_landscape_aggregator.stop()

    async def exec_algorithm(
        self,
        # request_id: str,
        algorithm_name: str,
        algorithm_request_dto: AlgorithmRequestDTO,
    ) -> str:

        context_id = UIDGenerator().get_a_uid()
        algo_execution_logger = ctrl_logger.get_request_logger(
            request_id=algorithm_request_dto.request_id
        )

        data_model = algorithm_request_dto.inputdata.data_model
        datasets = algorithm_request_dto.inputdata.datasets

        # instantiate nodes
        nodes = _create_nodes(
            request_id=algorithm_request_dto.request_id,
            context_id=context_id,
            data_model=data_model,
            datasets=datasets,
        )

        # add contextid and nodeids to cleaner
        local_nodes_ids = [node.node_id for node in nodes.local_nodes]
        all_nodes_ids = [nodes.global_node.node_id] + local_nodes_ids
        self._cleaner.add_contextid_for_cleanup(context_id, all_nodes_ids)

        # get metadata
        variable_names = (algorithm_request_dto.inputdata.x or []) + (
            algorithm_request_dto.inputdata.y or []
        )
        metadata = get_metadata(data_model=data_model, variable_names=variable_names)

        # create AlgorithmDTO
        algorithm_dto = AlgorithmDTO(
            algorithm_name=algorithm_name,
            data_model=algorithm_request_dto.inputdata.data_model,
            variables=Variables(
                x=sanitize_request_variable(algorithm_request_dto.inputdata.x),
                y=sanitize_request_variable(algorithm_request_dto.inputdata.y),
            ),
            var_filters=algorithm_request_dto.inputdata.filters,
            algorithm_parameters=algorithm_request_dto.parameters,
            metadata=metadata,
        )

        # instantiate algorithm
        algorithm = algorithm_classes[algorithm_name](algorithm_dto=algorithm_dto)

        command_id_generator = CommandIdGenerator()

        # create data model views
        data_model_views = _create_data_model_views(
            local_nodes=nodes.local_nodes,
            datasets=datasets,
            data_model=data_model,
            variable_groups=algorithm.get_variable_groups(),
            var_filters=algorithm_request_dto.inputdata.filters,
            dropna=algorithm.get_dropna(),
            check_min_rows=algorithm.get_check_min_rows(),
            command_id=command_id_generator.get_next_command_id(),
        )

        local_nodes_filtered = _filter_insufficient_data_nodes(
            nodes.local_nodes, data_model_views
        )
        nodes = Nodes(global_node=nodes.global_node, local_nodes=local_nodes_filtered)
        # check there are nodes left...
        if not nodes.local_nodes:
            raise RequestConstraintsError(algorithm_request_dto)

        # instantiate executor
        algorithm_executor_dto = AlgorithmExecutorDTO(
            request_id=algorithm_request_dto.request_id,
            context_id=context_id,
            algo_flags=algorithm_request_dto.flags,
            data_model_views=data_model_views,
        )
        algorithm_executor = _create_algorithm_executor(
            algorithm_executor_dto=algorithm_executor_dto,
            command_id_generator=command_id_generator,
            nodes=nodes,
        )

        log_experiment_execution(
            logger=algo_execution_logger,
            request_id=algorithm_request_dto.request_id,
            context_id=context_id,
            algorithm_name=algorithm_name,
            datasets=algorithm_request_dto.inputdata.datasets,
            algorithm_parameters=algorithm_request_dto.json(),
            local_node_ids=local_nodes_ids,
        )

        # run the algorithm
        try:
            # algorithm_result = algorithm.run(algorithm_executor)
            algorithm_result = await self._run_algorithm(
                algorithm=algorithm, algorithm_executor=algorithm_executor
            )

        except CeleryConnectionError as exc:
            algo_execution_logger.error(
                f"ErrorType: '{type(exc)}' and message: '{exc}'"
            )
            raise NodeUnresponsiveAlgorithmExecutionException()
        except CeleryTaskTimeoutException as exc:
            algo_execution_logger.error(
                f"ErrorType: '{type(exc)}' and message: '{exc}'"
            )
            raise NodeTaskTimeoutAlgorithmExecutionException()
        except Exception as exc:
            algo_execution_logger.error(traceback.format_exc())
            raise exc
        finally:
            self._cleaner.release_context_id(context_id=context_id)

        algo_execution_logger.info(
            f"Finished execution->  {algorithm_name=} with {algorithm_request_dto.request_id=}"
        )
        algo_execution_logger.debug(
            f"Algorithm {algorithm_request_dto.request_id=} result-> {algorithm_result.json()=}"
        )
        return algorithm_result.json()

    async def _run_algorithm(self, algorithm, algorithm_executor):
        # By calling blocking method Algorithm.run() inside run_in_executor(),
        # Algorithm.run() will execute in a separate thread of the threadpool and at
        # the same time yield control to the executor event loop, through await
        loop = asyncio.get_event_loop()
        algorithm_result = await loop.run_in_executor(
            self._executor, algorithm.run, algorithm_executor
        )
        return algorithm_result

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
        # return node_landscape_aggregator.get_datasets_locations()
        return self._node_landscape_aggregator.get_datasets_locations()

    def get_cdes_per_data_model(self) -> dict:
        return {
            data_model: {
                column: metadata.dict() for column, metadata in cdes.values.items()
            }
            # for data_model, cdes in node_landscape_aggregator.get_cdes_per_data_model().data_models_cdes.items()
            for data_model, cdes in self._node_landscape_aggregator.get_cdes_per_data_model().data_models_cdes.items()
        }

    def get_data_models_attributes(self) -> Dict[str, Dict]:
        return {
            data_model: data_model_metadata.dict()
            # for data_model, data_model_metadata in node_landscape_aggregator.get_data_models_attributes().items()
            for data_model, data_model_metadata in self._node_landscape_aggregator.get_data_models_attributes().items()
        }

    def get_all_available_data_models(self) -> List[str]:
        # return list(
        #     node_landscape_aggregator.get_cdes_per_data_model().data_models_cdes.keys()
        # )
        return list(
            self._node_landscape_aggregator.get_cdes_per_data_model().data_models_cdes.keys()
        )

    def get_all_available_datasets_per_data_model(self) -> Dict[str, List[str]]:
        # return node_landscape_aggregator.get_all_available_datasets_per_data_model()
        return (
            self._node_landscape_aggregator.get_all_available_datasets_per_data_model()
        )

    def get_all_local_nodes(self) -> List[NodeInfo]:
        # return node_landscape_aggregator.get_all_local_nodes()
        return self._node_landscape_aggregator.get_all_local_nodes()

    def get_global_node(self) -> NodeInfo:
        # return node_landscape_aggregator.get_global_node()
        return self._node_landscape_aggregator.get_global_node()

    def _get_node_info_by_id(self, node_id: str) -> _NodeInfoDTO:
        # node = node_landscape_aggregator.get_node_info(node_id)
        node = self._node_landscape_aggregator.get_node_info(node_id)

        return _NodeInfoDTO(
            node_id=node.id,
            queue_address=":".join([str(node.ip), str(node.port)]),
            db_address=":".join([str(node.db_ip), str(node.db_port)]),
            tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
            run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
        )


node_landscape_aggregator = NodeLandscapeAggregator()


def get_metadata(data_model: str, variable_names: List[str]):
    common_data_elements = node_landscape_aggregator.get_cdes(data_model)
    # common_data_elements = self._node_landscape_aggregator.get_cdes(data_model)
    metadata = {
        variable_name: cde.dict()
        for variable_name, cde in common_data_elements.items()
        if variable_name in variable_names
    }
    return metadata


def _create_data_model_views(
    local_nodes: List[LocalNode],
    datasets: List[str],
    data_model: str,
    variable_groups: List[List[str]],
    var_filters: list,
    dropna: bool,
    check_min_rows: bool,
    command_id: int,
) -> List[LocalNodesTable]:

    print("---------------------------------------------------------------")
    print(f"{local_nodes=}")
    print(f"{datasets=}")
    print(f"{data_model=}")
    print(f"{variable_groups=}")
    print(f"{var_filters=}")
    print(f"{dropna=}")
    print(f"{check_min_rows=}")
    print(f"{command_id=}")
    print("---------------------------------------------------------------")

    """
    Creates the data model views, for each variable group provided,
    using also the algorithm request arguments (data_model, datasets, filters).

    Parameters
    ----------
    variable_groups : List[List[str]]
        A list of variable_groups. The variable group is a list of columns.
    dropna : bool
        If True the view will not contain Remove NAs from the view.
    check_min_rows : bool
        Raise an exception if there are not enough rows in the view.

    Returns
    ------
    List[LocalNodesTable]
        A (LocalNodesTable) view for each variable_group provided.
    """
    # NOTE:there is something redundant here, the nodes have already been chosen because
    # they contain the specified data_model and datasets, so getting again the
    # data model and datasets of each node is redundant
    datasets_per_local_node = {
        local_node.node_id: node_landscape_aggregator.get_node_specific_datasets(
            # local_node.node_id: self._node_landscape_aggregator.get_node_specific_datasets(
            local_node.node_id,
            data_model,
            datasets,
        )
        # for node_id in local_nodes_ids
        for local_node in local_nodes
    }
    # breakpoint()
    command_id = str(command_id)  # TODO is str cast needed?
    views_per_localnode = []
    nodes_with_insuffiecient_data = []
    for node in local_nodes:
        try:
            data_model_views = node.create_data_model_views(
                command_id=command_id,
                data_model=data_model,
                datasets=datasets_per_local_node[node.node_id],
                columns_per_view=variable_groups,
                filters=var_filters,
                dropna=dropna,
                check_min_rows=check_min_rows,
            )
        except InsufficientDataError:
            continue
        views_per_localnode.append((node, data_model_views))

    if views_per_localnode:
        return _convert_views_per_localnode_to_local_nodes_tables(views_per_localnode)
    else:
        return []


def _filter_insufficient_data_nodes(
    local_nodes, data_model_views: List[LocalNodesTable]
):
    valid_nodes = {
        node
        for data_model_view in data_model_views
        for node in data_model_view.nodes_tables_info.keys()
    }

    tmp = [node for node in local_nodes if node in valid_nodes]

    if not tmp:
        raise InsufficientDataError(
            "None of the nodes has enough data to execute the "
            "algorithm. Algorithm with context_id="
            # f"{self._context_id} is aborted"
        )

    elif local_nodes != tmp:
        diff = set(local_nodes) - set(tmp)
        local_nodes = tmp

        # self._logger.info(
        #     f"Removed nodes:{diff} from algorithm with "
        #     f"context_id:{self._context_id}, because at least "
        #     f"one of the 'data model views' created on each of these nodes "
        #     f"contained insufficient rows. The algorithm will continue "
        #     f"executing on nodes: {self._local_nodes}"
        # )
    return local_nodes


def _create_nodes(request_id, context_id, data_model, datasets):
    node_tasks_handlers = _create_nodes_tasks_handlers(
        data_model=data_model, datasets=datasets
    )
    nodes = Nodes(
        global_node=_create_global_node(
            request_id, context_id, node_tasks_handlers.global_node_tasks_handler
        ),
        local_nodes=_create_local_nodes(
            request_id, context_id, node_tasks_handlers.local_nodes_tasks_handlers
        ),
    )

    return nodes


def _create_local_nodes(request_id, context_id, nodes_tasks_handlers):
    local_nodes: List[LocalNode] = [
        LocalNode(
            request_id=request_id,
            context_id=context_id,
            node_tasks_handler=node_tasks_handler,
        )
        for node_tasks_handler in nodes_tasks_handlers
    ]
    return local_nodes


def _create_global_node(request_id, context_id, node_tasks_handler):
    global_node: GlobalNode = GlobalNode(
        request_id=request_id,
        context_id=context_id,
        node_tasks_handler=node_tasks_handler,
    )
    return global_node


def _get_nodes_info_by_dataset(
    data_model: str, datasets: List[str]
) -> List[_NodeInfoDTO]:
    local_node_ids = node_landscape_aggregator.get_node_ids_with_any_of_datasets(
        # local_node_ids = self._node_landscape_aggregator.get_node_ids_with_any_of_datasets(
        data_model=data_model,
        datasets=datasets,
    )
    local_nodes_info = [
        node_landscape_aggregator.get_node_info(node_id)
        for node_id in local_node_ids
        # self._node_landscape_aggregator.get_node_info(node_id) for node_id in local_node_ids
    ]
    nodes_info = []
    for local_node in local_nodes_info:
        nodes_info.append(
            _NodeInfoDTO(
                node_id=local_node.id,
                queue_address=":".join([str(local_node.ip), str(local_node.port)]),
                db_address=":".join([str(local_node.db_ip), str(local_node.db_port)]),
                tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
                run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
            )
        )

    return nodes_info


def _create_nodes_tasks_handlers(
    data_model: str, datasets: List[str]
) -> NodesTasksHandlersDTO:
    # Get only the relevant nodes
    local_nodes_info = _get_nodes_info_by_dataset(
        data_model=data_model, datasets=datasets
    )
    local_nodes_tasks_handlers = [
        NodeAlgorithmTasksHandler(
            node_id=node_info.node_id,
            node_queue_addr=node_info.queue_address,
            node_db_addr=node_info.db_address,
            tasks_timeout=node_info.tasks_timeout,
            run_udf_task_timeout=node_info.run_udf_task_timeout,
        )
        for node_info in local_nodes_info
    ]

    global_node = node_landscape_aggregator.get_global_node()
    # global_node = self._node_landscape_aggregator.get_global_node()

    global_node_tasks_handler = NodeAlgorithmTasksHandler(
        node_id=global_node.id,
        node_queue_addr=":".join([str(global_node.ip), str(global_node.port)]),
        node_db_addr=":".join([str(global_node.db_ip), str(global_node.db_port)]),
        tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
        run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
    )

    return NodesTasksHandlersDTO(
        global_node_tasks_handler=global_node_tasks_handler,
        local_nodes_tasks_handlers=local_nodes_tasks_handlers,
    )


def _convert_views_per_localnode_to_local_nodes_tables(
    views_per_localnode: List[Tuple[LocalNode, List[TableInfo]]]
) -> List[LocalNodesTable]:
    """
    In the views_per_localnode the views are stored per the localnode where they exist.
    In order to create LocalNodesTable objects we need to store them according to the similar "LocalNodesTable"
    they belong to. We group together one view from each node, based on the views' order.

    Parameters
    ----------
    views_per_localnode: views grouped per the localnode where they exist.

    Returns
    ------
    One (LocalNodesTable) view for each one existing in the localnodes.
    """
    views_count = _get_amount_of_localnodes_views(views_per_localnode)

    local_nodes_tables_dicts: List[Dict[LocalNode, TableInfo]] = [
        {} for _ in range(views_count)
    ]
    for localnode, local_node_views in views_per_localnode:
        for view, local_nodes_tables in zip(local_node_views, local_nodes_tables_dicts):
            local_nodes_tables[localnode] = view
    local_nodes_tables = [
        LocalNodesTable(local_nodes_tables_dict)
        for local_nodes_tables_dict in local_nodes_tables_dicts
    ]
    return local_nodes_tables


def _get_amount_of_localnodes_views(
    views_per_localnode: List[Tuple[LocalNode, List[TableInfo]]]
) -> int:
    """
    Returns the amount of views after validating all localnodes created the same amount of views.
    """
    views_count = len(views_per_localnode[0][1])
    for local_node, local_node_views in views_per_localnode:
        if len(local_node_views) != views_count:
            raise ValueError(
                f"All views from localnodes should have the same length. "
                f"{local_node} has {len(local_node_views)} instead of {views_count}."
            )
    return views_count


def _create_algorithm_executor(
    algorithm_executor_dto: AlgorithmExecutorDTO,
    command_id_generator: CommandIdGenerator,
    nodes: Nodes,
):
    if len(nodes.local_nodes) < 2:
        return AlgorithmExecutorSingleLocalNode(
            algorithm_executor_dto=algorithm_executor_dto,
            command_id_generator=command_id_generator,
            nodes=nodes,
        )
    else:
        return AlgorithmExecutor(
            algorithm_executor_dto=algorithm_executor_dto,
            command_id_generator=command_id_generator,
            nodes=nodes,
        )


def sanitize_request_variable(variable: list):
    if variable:
        return variable
    else:
        return []
