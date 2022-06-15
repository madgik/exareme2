import threading
import time
import traceback
from threading import Lock
from typing import Dict
from typing import List
from typing import Tuple

from mipengine.controller import config as controller_config
from mipengine.controller import controller_logger as ctrl_logger
from mipengine.controller.celery_app import CeleryConnectionError
from mipengine.controller.celery_app import CeleryTaskTimeoutException
from mipengine.controller.data_model_registry import DataModelRegistry
from mipengine.controller.node_info_tasks_handler import NodeInfoTasksHandler
from mipengine.controller.node_registry import NodeRegistry
from mipengine.controller.nodes_addresses import get_nodes_addresses
from mipengine.node_info_DTOs import NodeInfo
from mipengine.node_info_DTOs import NodeRole
from mipengine.node_tasks_DTOs import CommonDataElement
from mipengine.node_tasks_DTOs import CommonDataElements
from mipengine.singleton import Singleton

logger = ctrl_logger.get_background_service_logger()

NODE_LANDSCAPE_AGGREGATOR_REQUEST_ID = "NODE_LANDSCAPE_AGGREGATOR"
NODE_LANDSCAPE_AGGREGATOR_UPDATE_INTERVAL = (
    controller_config.node_landscape_aggregator_update_interval
)
CELERY_TASKS_TIMEOUT = controller_config.rabbitmq.celery_tasks_timeout


def _get_nodes_info(nodes_socket_addr: List[str]) -> List[NodeInfo]:
    node_info_tasks_handlers = [
        NodeInfoTasksHandler(
            node_queue_addr=node_socket_addr, tasks_timeout=CELERY_TASKS_TIMEOUT
        )
        for node_socket_addr in nodes_socket_addr
    ]

    nodes_info = []
    for tasks_handler in node_info_tasks_handlers:
        try:
            async_result = tasks_handler.queue_node_info_task(
                request_id=NODE_LANDSCAPE_AGGREGATOR_REQUEST_ID
            )
            result = tasks_handler.result_node_info_task(
                async_result=async_result,
                request_id=NODE_LANDSCAPE_AGGREGATOR_REQUEST_ID,
            )
            nodes_info.append(result)

        except (CeleryConnectionError, CeleryTaskTimeoutException) as exc:
            # just log the exception do not reraise it
            logger.warning(exc)
        except Exception as exc:
            # just log full traceback exception as error and do not reraise it
            logger.error(traceback.format_exc())
    return nodes_info


def _get_node_datasets_per_data_model(
    node_socket_addr: str,
) -> Dict[str, Dict[str, str]]:
    tasks_handler = NodeInfoTasksHandler(
        node_queue_addr=node_socket_addr, tasks_timeout=CELERY_TASKS_TIMEOUT
    )

    datasets_per_data_model = {}
    try:
        async_result = tasks_handler.queue_node_datasets_per_data_model_task(
            request_id=NODE_LANDSCAPE_AGGREGATOR_REQUEST_ID
        )
        datasets_per_data_model = (
            tasks_handler.result_node_datasets_per_data_model_task(
                async_result=async_result,
                request_id=NODE_LANDSCAPE_AGGREGATOR_REQUEST_ID,
            )
        )

    except (CeleryConnectionError, CeleryTaskTimeoutException) as exc:
        # just log the exception do not reraise it
        logger.warning(exc)
    except Exception as exc:
        # just log full traceback exception as error and do not reraise it
        logger.error(traceback.format_exc())

    return datasets_per_data_model


def _get_node_cdes(node_socket_addr: str, data_model: str) -> CommonDataElements:
    tasks_handler = NodeInfoTasksHandler(
        node_queue_addr=node_socket_addr, tasks_timeout=CELERY_TASKS_TIMEOUT
    )
    try:
        async_result = tasks_handler.queue_data_model_cdes_task(
            request_id=NODE_LANDSCAPE_AGGREGATOR_REQUEST_ID, data_model=data_model
        )
        datasets_per_data_model = tasks_handler.result_data_model_cdes_task(
            async_result=async_result, request_id=NODE_LANDSCAPE_AGGREGATOR_REQUEST_ID
        )
        return datasets_per_data_model
    except (CeleryConnectionError, CeleryTaskTimeoutException) as exc:
        # just log the exception do not reraise it
        logger.warning(exc)
    except Exception as exc:
        # just log full traceback exception as error and do not reraise it
        logger.error(traceback.format_exc())


def _get_node_socket_addr(node_info: NodeInfo):
    return f"{node_info.ip}:{node_info.port}"


class NodeLandscapeAggregator(metaclass=Singleton):
    def __init__(self):
        self._initialize()

        self._keep_updating = True
        self._update_loop_thread = None
        self._update_lock = Lock()

    def _update(self):

        """
        Node Landscape Aggregator(NLA) is a module that handles the aggregation of necessary information,
        to keep up-to-date and in sync the Node Registry and the Data Model Registry.
        The Node Registry contains information about the node such as id, ip, port etc.
        The Data Model Registry contains two types of information, data_models and datasets_location.
        data_models contains information about the data models and their corresponding cdes.
        datasets_location contains information about datasets and their locations(nodes).
        NLA periodically will send requests (get_node_info, get_node_datasets_per_data_model, get_data_model_cdes),
        to the nodes to retrieve the current information that they contain.
        Once all information about data models and cdes is aggregated,
        any data model that is incompatible across nodes will be removed.
        A data model is incompatible when the cdes across nodes are not identical, except one edge case.
        The edge case is that the cdes can only contain a difference in the field of 'enumerations' in
        the cde with code 'dataset' and still be considered compatible.
        For each data model the 'enumerations' field in the cde with code 'dataset' is updated with all datasets across nodes.
        Once all the information is aggregated and validated the NLA will provide the information to the Node Registry and to the Data Model Registry.
        """
        while self._keep_updating:
            try:
                nodes_addresses = get_nodes_addresses()
                nodes_info = _get_nodes_info(nodes_addresses)

                with self._update_lock:
                    self._node_registry.nodes = {
                        node_info.id: node_info for node_info in nodes_info
                    }

                    local_nodes = [
                        node for node in nodes_info if node.role == NodeRole.LOCALNODE
                    ]
                    (
                        dataset_locations,
                        aggregated_datasets,
                    ) = _gather_all_dataset_info(local_nodes)
                    data_model_cdes_per_node = _get_cdes_across_nodes(local_nodes)
                    compatible_data_models = _get_compatible_data_models(
                        data_model_cdes_per_node
                    )
                    _update_data_models_with_aggregated_datasets(
                        compatible_data_models, aggregated_datasets
                    )
                    datasets_locations = (
                        _get_dataset_locations_of_compatible_data_models(
                            compatible_data_models, dataset_locations
                        )
                    )

                    self._data_model_registry.data_models = compatible_data_models
                    self._data_model_registry.datasets_location = datasets_locations

                    logger.info(
                        f"Online nodes:{[node for node in self._node_registry.nodes]}"
                    )

            except Exception as exc:
                logger.warning(
                    f"NodeLandscapeAggregator caught an exception but will continue to "
                    f"update {exc=}"
                )
            finally:
                time.sleep(NODE_LANDSCAPE_AGGREGATOR_UPDATE_INTERVAL)

    def _initialize(self):
        self._node_registry = NodeRegistry(logger)
        self._data_model_registry = DataModelRegistry(logger)

    def start(self):
        self.stop()

        self._initialize()
        self._keep_updating = True

        self._update_loop_thread = threading.Thread(target=self._update, daemon=True)
        self._update_loop_thread.start()

    def stop(self):
        if self._update_loop_thread and self._update_loop_thread.is_alive():
            self._keep_updating = False
            self._update_loop_thread.join()

    def get_nodes(self) -> Dict[str, NodeInfo]:
        with self._update_lock:
            return self._node_registry.nodes

    def get_global_node(self) -> NodeInfo:
        with self._update_lock:
            return self._node_registry.get_global_node()

    def get_all_local_nodes(self) -> Dict[str, NodeInfo]:
        with self._update_lock:
            return self._node_registry.get_all_local_nodes()

    def get_node_info(self, node_id: str) -> NodeInfo:
        with self._update_lock:
            return self._node_registry.get_node_info(node_id)

    def get_cdes(self, data_model: str) -> Dict[str, CommonDataElement]:
        with self._update_lock:
            return self._data_model_registry.get_cdes(data_model)

    def get_cdes_per_data_model(self) -> Dict[str, CommonDataElements]:
        with self._update_lock:
            return self._data_model_registry.data_models

    def get_datasets_location(self) -> Dict[str, Dict[str, List[str]]]:
        with self._update_lock:
            return self._data_model_registry.datasets_location

    def get_all_available_datasets_per_data_model(self) -> Dict[str, List[str]]:
        with self._update_lock:
            return self._data_model_registry.get_all_available_datasets_per_data_model()

    def data_model_exists(self, data_model: str) -> bool:
        with self._update_lock:
            return self._data_model_registry.data_model_exists(data_model)

    def dataset_exists(self, data_model: str, dataset: str) -> bool:
        with self._update_lock:
            return self._data_model_registry.dataset_exists(data_model, dataset)

    def get_node_ids_with_any_of_datasets(
        self, data_model: str, datasets: List[str]
    ) -> List[str]:
        with self._update_lock:
            return self._data_model_registry.get_node_ids_with_any_of_datasets(
                data_model, datasets
            )

    def get_node_specific_datasets(
        self, node_id: str, data_model: str, wanted_datasets: List[str]
    ) -> List[str]:
        with self._update_lock:
            return self._data_model_registry.get_node_specific_datasets(
                node_id, data_model, wanted_datasets
            )


def _gather_all_dataset_info(
    nodes: List[NodeInfo],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    """

    Args:
        nodes: The nodes available in the system

    Returns:
        A tuple with:
         1. The location of each dataset.
         2. The aggregated datasets, existing in all nodes
    """
    dataset_locations = {}
    aggregated_datasets = {}

    for node_info in nodes:
        node_socket_addr = _get_node_socket_addr(node_info)
        datasets_per_data_model = _get_node_datasets_per_data_model(node_socket_addr)
        for data_model, datasets in datasets_per_data_model.items():
            current_labels = (
                aggregated_datasets[data_model]
                if data_model in aggregated_datasets
                else {}
            )
            current_datasets = (
                dataset_locations[data_model] if data_model in dataset_locations else {}
            )

            for dataset in datasets:
                current_labels[dataset] = datasets[dataset]

                if dataset in current_datasets:
                    current_datasets[dataset].append(node_info.id)
                else:
                    current_datasets[dataset] = [node_info.id]

            aggregated_datasets[data_model] = current_labels
            dataset_locations[data_model] = current_datasets
    return dataset_locations, aggregated_datasets


def _get_cdes_across_nodes(
    nodes: List[NodeInfo],
) -> Dict[str, List[Tuple[str, CommonDataElements]]]:
    nodes_cdes = {}
    for node_info in nodes:
        node_socket_addr = _get_node_socket_addr(node_info)
        datasets_per_data_model = _get_node_datasets_per_data_model(node_socket_addr)
        for data_model in datasets_per_data_model:
            cdes = _get_node_cdes(node_socket_addr, data_model)
            if cdes:
                if data_model not in nodes_cdes:
                    nodes_cdes[data_model] = []
                    nodes_cdes[data_model].append((node_info.id, cdes))
    return nodes_cdes


def _get_dataset_locations_of_compatible_data_models(
    compatible_data_models, dataset_locations
):
    return {
        compatible_data_model: dataset_locations[compatible_data_model]
        for compatible_data_model in compatible_data_models
    }


def _get_compatible_data_models(
    data_model_cdes_across_nodes: Dict[str, List[Tuple[str, CommonDataElements]]]
) -> Dict[str, CommonDataElements]:
    """
    Each node has its own data models definition.
    We need to check for each data model if the definitions across all nodes is the same.
    If the data model is not the same across all nodes containing it, we log the incompatibility.
    The data models with similar definitions across all nodes are returned.

    Parameters
    ----------
        data_model_cdes_across_nodes: the data_models each node has

    Returns
    ----------
        Dict[str, CommonDataElements]
            the data models with similar definitions across all nodes

    """
    data_models = {}
    for data_model, cdes_from_all_nodes in data_model_cdes_across_nodes.items():
        first_node, first_cdes = cdes_from_all_nodes[0]
        for node, cdes in cdes_from_all_nodes[1:]:
            if not first_cdes == cdes:
                logger.info(
                    f"Node '{first_node}' and node '{node}' on data model '{data_model}' have incompatibility on the "
                    f"following cdes: {first_cdes} and {cdes} "
                )
                break
        else:
            data_models[data_model] = first_cdes

    return data_models


def _update_data_models_with_aggregated_datasets(
    data_models: Dict[str, CommonDataElements],
    aggregated_datasets: Dict[str, Dict[str, str]],
):
    """
    Updates each data_model's 'dataset' enumerations with the aggregated datasets
    """
    for data_model in data_models:
        dataset_cde = data_models[data_model].values["dataset"]
        new_dataset_cde = CommonDataElement(
            code=dataset_cde.code,
            label=dataset_cde.label,
            sql_type=dataset_cde.sql_type,
            is_categorical=dataset_cde.is_categorical,
            enumerations=aggregated_datasets[data_model],
            min=dataset_cde.min,
            max=dataset_cde.max,
        )
        data_models[data_model].values["dataset"] = new_dataset_cde
