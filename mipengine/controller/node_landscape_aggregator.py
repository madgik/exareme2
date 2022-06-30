import threading
import time
import traceback
from collections import Counter
from typing import Dict
from typing import List
from typing import Tuple

from pydantic import BaseModel

from mipengine.controller import config as controller_config
from mipengine.controller import controller_logger as ctrl_logger
from mipengine.controller.celery_app import CeleryConnectionError
from mipengine.controller.celery_app import CeleryTaskTimeoutException
from mipengine.controller.data_model_registry import DataModelRegistry
from mipengine.controller.federation_info_logs import log_datamodel_added
from mipengine.controller.federation_info_logs import log_datamodel_removed
from mipengine.controller.federation_info_logs import log_dataset_added
from mipengine.controller.federation_info_logs import log_dataset_removed
from mipengine.controller.federation_info_logs import log_node_joined_federation
from mipengine.controller.federation_info_logs import log_node_left_federation
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
        return {}
    except Exception as exc:
        # just log full traceback exception as error and do not reraise it
        logger.error(traceback.format_exc())
        return {}

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


class _NLARegistries(BaseModel):
    node_registry: NodeRegistry
    data_model_registry: DataModelRegistry

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True


class NodeLandscapeAggregator(metaclass=Singleton):
    def __init__(self):
        self._registries = _NLARegistries(
            node_registry=NodeRegistry(nodes={}),
            data_model_registry=DataModelRegistry(data_models={}, dataset_location={}),
        )

        self._keep_updating = True
        self._update_loop_thread = None

    def _update(self):

        """
        Node Landscape Aggregator(NLA) is a module that handles the aggregation of necessary information,
        to keep up-to-date and in sync the Node Registry and the Data Model Registry.
        The Node Registry contains information about the node such as id, ip, port etc.
        The Data Model Registry contains two types of information, data_models and dataset_location.
        data_models contains information about the data models and their corresponding cdes.
        dataset_location contains information about datasets and their locations(nodes).
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
                local_nodes = {
                    node_info.id: node_info
                    for node_info in nodes_info
                    if node_info.role == NodeRole.LOCALNODE
                }

                datasets_per_node = _get_datasets_per_node(local_nodes)

                data_model_cdes_per_node = _get_cdes_across_nodes(
                    local_nodes, datasets_per_node
                )
                compatible_data_models = _get_compatible_data_models(
                    data_model_cdes_per_node
                )

                (
                    dataset_locations,
                    aggregated_datasets,
                ) = _gather_all_dataset_info(datasets_per_node)
                _update_data_models_with_aggregated_datasets(
                    data_models=compatible_data_models,
                    aggregated_datasets=aggregated_datasets,
                )

                dataset_locations = _get_dataset_locations_of_compatible_data_models(
                    compatible_data_models, dataset_locations
                )

                nodes = {node_info.id: node_info for node_info in nodes_info}

                self.set_new_registy_values(
                    nodes, compatible_data_models, dataset_locations
                )

                logger.debug(
                    f"Nodes:{[node for node in self._registries.node_registry.nodes]}"
                )

            except Exception as exc:
                logger.warning(
                    f"NodeLandscapeAggregator caught an exception but will continue to "
                    f"update {exc=}"
                )

                tr = traceback.format_exc()
                logger.error(tr)
            finally:
                time.sleep(NODE_LANDSCAPE_AGGREGATOR_UPDATE_INTERVAL)

    def start(self):
        self.stop()

        self._keep_updating = True

        self._update_loop_thread = threading.Thread(target=self._update, daemon=True)
        self._update_loop_thread.start()

    def stop(self):
        if self._update_loop_thread and self._update_loop_thread.is_alive():
            self._keep_updating = False
            self._update_loop_thread.join()

    def set_new_registy_values(self, nodes, data_models, dataset_location):
        log_node_changes(logger, self._registries.node_registry.nodes, nodes)
        log_data_model_changes(
            logger,
            self._registries.data_model_registry.data_models,
            data_models,
        )
        log_dataset_changes(
            logger,
            self._registries.data_model_registry.dataset_location,
            dataset_location,
        )
        _node_registry = NodeRegistry(nodes=nodes)
        _data_model_registry = DataModelRegistry(
            data_models=data_models,
            dataset_location=dataset_location,
        )

        self._registries = _NLARegistries(
            node_registry=_node_registry, data_model_registry=_data_model_registry
        )

    def get_nodes(self) -> Dict[str, NodeInfo]:
        return self._registries.node_registry.nodes

    def get_global_node(self) -> NodeInfo:
        return self._registries.node_registry.get_global_node()

    def get_all_local_nodes(self) -> Dict[str, NodeInfo]:
        return self._registries.node_registry.get_all_local_nodes()

    def get_node_info(self, node_id: str) -> NodeInfo:
        return self._registries.node_registry.get_node_info(node_id)

    def get_cdes(self, data_model: str) -> Dict[str, CommonDataElement]:
        return self._registries.data_model_registry.get_cdes(data_model)

    def get_cdes_per_data_model(self) -> Dict[str, CommonDataElements]:
        return self._registries.data_model_registry.data_models

    def get_dataset_location(self) -> Dict[str, Dict[str, List[str]]]:
        return self._registries.data_model_registry.dataset_location

    def get_all_available_datasets_per_data_model(self) -> Dict[str, List[str]]:
        return (
            self._registries.data_model_registry.get_all_available_datasets_per_data_model()
        )

    def data_model_exists(self, data_model: str) -> bool:
        return self._registries.data_model_registry.data_model_exists(data_model)

    def dataset_exists(self, data_model: str, dataset: str) -> bool:
        return self._registries.data_model_registry.dataset_exists(data_model, dataset)

    def get_node_ids_with_any_of_datasets(
        self, data_model: str, datasets: List[str]
    ) -> List[str]:
        return self._registries.data_model_registry.get_node_ids_with_any_of_datasets(
            data_model, datasets
        )

    def get_node_specific_datasets(
        self, node_id: str, data_model: str, wanted_datasets: List[str]
    ) -> List[str]:
        return self._registries.data_model_registry.get_node_specific_datasets(
            node_id, data_model, wanted_datasets
        )


def _get_datasets_per_node(
    nodes: Dict[str, NodeInfo],
) -> Dict[str, Dict[str, Dict[str, str]]]:
    datasets_per_node = {}
    for node_id, node_info in nodes.items():
        node_socket_addr = _get_node_socket_addr(node_info)
        datasets_per_data_model = _get_node_datasets_per_data_model(node_socket_addr)
        if datasets_per_data_model:
            datasets_per_node[node_info.id] = datasets_per_data_model

    return datasets_per_node


def remove_duplicated_datasets(datasets_per_node):
    aggregated_datasets = [
        dataset
        for datasets_per_data_model in datasets_per_node.values()
        for datasets in datasets_per_data_model.values()
        for dataset in datasets
    ]
    duplicated = [
        item for item, count in Counter(aggregated_datasets).items() if count > 1
    ]
    for node, datasets_per_data_model in datasets_per_node.items():
        for data_model, datasets in datasets_per_data_model.items():
            datasets_per_node[node][data_model] = {
                dataset_name: dataset_label
                for dataset_name, dataset_label in datasets.items()
                if dataset_name not in duplicated
            }


def _gather_all_dataset_info(
    datasets_per_node: Dict[str, Dict[str, Dict[str, str]]],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    """

    Args:
        datasets_per_node: The datasets for each node available in the system

    Returns:
        A tuple with:
         1. The location of each dataset.
         2. The aggregated datasets, existing in all nodes
    """
    dataset_locations = {}
    aggregated_datasets = {}

    for node_id, datasets_per_data_model in datasets_per_node.items():

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
                current_datasets[dataset] = node_id

            aggregated_datasets[data_model] = current_labels
            dataset_locations[data_model] = current_datasets
    return dataset_locations, aggregated_datasets


def _get_cdes_across_nodes(
    nodes: Dict[str, NodeInfo],
    datasets_per_node: Dict[str, Dict[str, Dict[str, str]]],
) -> Dict[str, List[Tuple[str, CommonDataElements]]]:
    nodes_cdes = {}
    for node_id, datasets_per_data_model in datasets_per_node.items():
        node_socket_addr = _get_node_socket_addr(nodes[node_id])
        for data_model in datasets_per_data_model:
            cdes = _get_node_cdes(node_socket_addr, data_model)
            if not cdes:
                del datasets_per_node[node_id][data_model]
                continue
            cdes = _get_node_cdes(node_socket_addr, data_model)
            if data_model not in nodes_cdes:
                nodes_cdes[data_model] = []
            nodes_cdes[data_model].append((node_id, cdes))
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
                    f"Node '{first_node}' and node '{node}' on data model '{data_model}' "
                    f"have incompatibility on the following cdes: {first_cdes} and {cdes} "
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
        if data_models[data_model]:
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


def log_node_changes(_logger, old_nodes, new_nodes):
    added_nodes = set(new_nodes.keys()) - set(old_nodes.keys())
    for node in added_nodes:
        log_node_joined_federation(_logger, node)

    removed_nodes = set(old_nodes.keys()) - set(new_nodes.keys())
    for node in removed_nodes:
        log_node_left_federation(_logger, node)


def log_data_model_changes(_logger, old_data_models, new_data_models):
    added_data_models = new_data_models.keys() - old_data_models.keys()
    for data_model in added_data_models:
        log_datamodel_added(data_model, _logger)

    removed_data_models = old_data_models.keys() - new_data_models.keys()
    for data_model in removed_data_models:
        log_datamodel_removed(data_model, _logger)


def log_dataset_changes(
    _logger, old_datasets_per_data_model, new_datasets_per_data_model
):
    _log_datasets_added(
        _logger, old_datasets_per_data_model, new_datasets_per_data_model
    )
    _log_datasets_removed(
        _logger, old_datasets_per_data_model, new_datasets_per_data_model
    )


def _log_datasets_added(
    _logger, old_datasets_per_data_model, new_datasets_per_data_model
):
    for data_model in new_datasets_per_data_model:
        added_datasets = new_datasets_per_data_model[data_model].keys()
        if data_model in old_datasets_per_data_model:
            added_datasets -= old_datasets_per_data_model[data_model].keys()
        for dataset in added_datasets:
            log_dataset_added(data_model, dataset, _logger, new_datasets_per_data_model)


def _log_datasets_removed(
    _logger, old_datasets_per_data_model, new_datasets_per_data_model
):
    for data_model in old_datasets_per_data_model:
        removed_datasets = old_datasets_per_data_model[data_model].keys()
        if data_model in new_datasets_per_data_model:
            removed_datasets -= new_datasets_per_data_model[data_model].keys()
        for dataset in removed_datasets:
            log_dataset_removed(
                data_model, dataset, _logger, old_datasets_per_data_model
            )
