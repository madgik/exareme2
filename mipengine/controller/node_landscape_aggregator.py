import threading
import time
import traceback
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


class DatasetInfo(BaseModel):
    values: Dict[str, str]


class DatamodelInfo(BaseModel):
    values: Dict[str, Tuple[DatasetInfo, CommonDataElements]]


class FederationInfo(BaseModel):
    values: Dict[str, DatamodelInfo]


class _NLARegistries(BaseModel):
    node_registry: NodeRegistry
    data_model_registry: DataModelRegistry

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True


class NodeLandscapeAggregator(metaclass=Singleton):
    def __init__(self):
        self._registries = _NLARegistries(
            node_registry=NodeRegistry(),
            data_model_registry=DataModelRegistry(),
        )
        self._keep_updating = True
        self._update_loop_thread = None

    def _update(self):
        """
        Node Landscape Aggregator(NLA) is a module that handles the aggregation of necessary information,
        to keep up-to-date and in sync the Node Registry and the Data Model Registry.
        The Node Registry contains information about the node such as id, ip, port etc.
        The Data Model Registry contains two types of information, data_models and datasets_locations.
        data_models contains information about the data models and their corresponding cdes.
        datasets_locations contains information about datasets and their locations(nodes).
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
                (nodes_info, federated_node_infos) = data_fetching()
                logger.info(federated_node_infos.__str__())
                for node_id, fdmi in federated_node_infos.values.items():
                    for data_model, datasets_and_cdes in fdmi.values.items():
                        logger.info(f"{node_id=}\n")
                        logger.info(f"{data_model=}\n")
                        logger.info(f"{datasets_and_cdes[0]=}\n")
                node_registry = NodeRegistry(nodes_info=nodes_info)
                dmr = data_model_registry_data_cruncing(federated_node_infos)

                self.set_new_registries(node_registry, dmr)

                logger.debug(
                    f"Nodes:{[node_info.id for node_info in self._registries.node_registry.get_nodes()]}"
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

    def set_new_registries(self, node_registry, data_model_registry):
        _log_node_changes(
            self._registries.node_registry.get_nodes(),
            node_registry.get_nodes(),
        )
        _log_data_model_changes(
            self._registries.data_model_registry.data_models,
            data_model_registry.data_models,
        )
        _log_dataset_changes(
            self._registries.data_model_registry.datasets_locations,
            data_model_registry.datasets_locations,
        )
        self._registries = _NLARegistries(
            node_registry=node_registry, data_model_registry=data_model_registry
        )

    def get_nodes(self) -> List[NodeInfo]:
        return self._registries.node_registry.get_nodes()

    def get_global_node(self) -> NodeInfo:
        return self._registries.node_registry.global_node

    def get_all_local_nodes(self) -> List[NodeInfo]:
        return self._registries.node_registry.local_nodes

    def get_node_info(self, node_id: str) -> NodeInfo:
        return self._registries.node_registry.get_node_info(node_id)

    def get_cdes(self, data_model: str) -> Dict[str, CommonDataElement]:
        return self._registries.data_model_registry.get_cdes(data_model)

    def get_cdes_per_data_model(self) -> Dict[str, CommonDataElements]:
        return self._registries.data_model_registry.data_models

    def get_datasets_locations(self) -> Dict[str, Dict[str, str]]:
        return self._registries.data_model_registry.datasets_locations

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


def data_fetching() -> Tuple[
    List[NodeInfo],
    FederationInfo,
]:
    nodes_addresses = get_nodes_addresses()
    nodes_info = _get_nodes_info(nodes_addresses)
    local_nodes = [
        node_info for node_info in nodes_info if node_info.role == NodeRole.LOCALNODE
    ]
    federated_node_infos = _get_federated_node_infos(local_nodes)
    return nodes_info, federated_node_infos


def data_model_registry_data_cruncing(
    federated_node_infos: FederationInfo,
) -> DataModelRegistry:
    federated_node_infos_with_compatible_data_models = _remove_incompatible_data_models(
        federated_node_infos
    )
    datasets_locations, aggregated_datasets = _gather_all_dataset_info(
        federated_node_infos_with_compatible_data_models
    )
    data_models = _get_data_models(
        federated_node_infos_with_compatible_data_models, aggregated_datasets
    )

    return DataModelRegistry(
        data_models=data_models, datasets_locations=datasets_locations
    )


def _get_federated_node_infos(
    nodes: List[NodeInfo],
) -> FederationInfo:
    fni = {}

    for node_info in nodes:
        fdmi = {}

        node_socket_addr = _get_node_socket_addr(node_info)
        datasets_per_data_model = _get_node_datasets_per_data_model(node_socket_addr)
        if datasets_per_data_model:
            node_socket_addr = _get_node_socket_addr(node_info)
            for data_model, datasets in datasets_per_data_model.items():

                cdes = _get_node_cdes(node_socket_addr, data_model)
                if not cdes:
                    continue
                fdmi[data_model] = (DatasetInfo(values=datasets), cdes)
        fni[node_info.id] = DatamodelInfo(values=fdmi)

    return FederationInfo(values=fni)


def _gather_all_dataset_info(federated_node_infos):
    datasets_locations = {}
    aggregated_datasets = {}
    for node_id, fdmi in federated_node_infos.values.items():
        for data_model, datasets_and_cdes in fdmi.values.items():
            datasets, _ = datasets_and_cdes

            current_labels = (
                aggregated_datasets[data_model]
                if data_model in aggregated_datasets
                else {}
            )
            current_datasets = (
                datasets_locations[data_model]
                if data_model in datasets_locations
                else {}
            )

            for dataset_name, dataset_label in datasets.values.items():
                current_labels[dataset_name] = dataset_label

                if dataset_name in current_datasets:
                    current_datasets[dataset_name].append(node_id)
                else:
                    current_datasets[dataset_name] = [node_id]

            aggregated_datasets[data_model] = current_labels
            datasets_locations[data_model] = current_datasets

    datasets_locations_without_duplicates = {}
    for data_model, dataset_locations in datasets_locations.items():
        datasets_locations_without_duplicates[data_model] = {}

        for dataset, node_ids in dataset_locations.items():
            if len(node_ids) == 1:
                datasets_locations_without_duplicates[data_model][dataset] = node_ids[0]
            else:
                del aggregated_datasets[data_model][dataset]
                _log_duplicated_dataset(node_ids, data_model, dataset)

    return datasets_locations_without_duplicates, aggregated_datasets


def _get_data_models(federated_node_infos, aggregated_datasets):
    data_models = {}
    for node_id, fdmi in federated_node_infos.values.items():
        for data_model, datasets_and_cdes in fdmi.values.items():
            _, cdes = datasets_and_cdes
            data_models[data_model] = cdes
            dataset_cde = cdes.values["dataset"]
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
    return data_models


def _remove_incompatible_data_models(
    federated_node_infos: FederationInfo,
) -> FederationInfo:
    validation_dictionary = {}

    incompatible_data_models = []
    for node_id, fdmi in federated_node_infos.values.items():
        for data_model, datasets_and_cdes in fdmi.values.items():
            if data_model in incompatible_data_models:
                continue

            _, cdes = datasets_and_cdes
            if data_model in validation_dictionary:
                valid_node_id, valid_cdes = validation_dictionary[data_model]
                if not valid_cdes == cdes:
                    nodes = [node_id, valid_node_id]
                    incompatible_data_models.append(data_model)
                    _log_incompatible_data_models(nodes, data_model, [cdes, valid_cdes])
                    break
            else:
                validation_dictionary[data_model] = (node_id, cdes)

    compatible_federated_node_infos = {
        node_id: DatamodelInfo(
            values={
                data_model: datasets_and_cdes
                for data_model, datasets_and_cdes in fdmi.values.items()
                if data_model not in incompatible_data_models
            }
        )
        for node_id, fdmi in federated_node_infos.values.items()
    }
    return FederationInfo(values=compatible_federated_node_infos)


def _log_node_changes(old_nodes, new_nodes):
    old_nodes_per_node_id = {node.id: node for node in old_nodes}
    new_nodes_per_node_id = {node.id: node for node in new_nodes}
    added_nodes = set(new_nodes_per_node_id.keys()) - set(old_nodes_per_node_id.keys())
    for node in added_nodes:
        log_node_joined_federation(logger, node)
    removed_nodes = set(old_nodes_per_node_id.keys()) - set(
        new_nodes_per_node_id.keys()
    )
    for node in removed_nodes:
        log_node_left_federation(logger, node)


def _log_data_model_changes(old_data_models, new_data_models):
    added_data_models = new_data_models.keys() - old_data_models.keys()
    for data_model in added_data_models:
        log_datamodel_added(data_model, logger)
    removed_data_models = old_data_models.keys() - new_data_models.keys()
    for data_model in removed_data_models:
        log_datamodel_removed(data_model, logger)


def _log_dataset_changes(old_datasets_locations, new_datasets_locations):
    _log_datasets_added(old_datasets_locations, new_datasets_locations)
    _log_datasets_removed(old_datasets_locations, new_datasets_locations)


def _log_datasets_added(old_datasets_locations, new_datasets_locations):
    for data_model in new_datasets_locations:
        added_datasets = new_datasets_locations[data_model].keys()
        if data_model in old_datasets_locations:
            added_datasets -= old_datasets_locations[data_model].keys()
        for dataset in added_datasets:
            log_dataset_added(
                data_model,
                dataset,
                logger,
                new_datasets_locations[data_model][dataset],
            )


def _log_datasets_removed(old_datasets_locations, new_datasets_locations):
    for data_model in old_datasets_locations:
        removed_datasets = old_datasets_locations[data_model].keys()
        if data_model in new_datasets_locations:
            removed_datasets -= new_datasets_locations[data_model].keys()
        for dataset in removed_datasets:
            log_dataset_removed(
                data_model,
                dataset,
                logger,
                old_datasets_locations[data_model][dataset],
            )


def _log_incompatible_data_models(nodes, data_model, conflicting_cdes):
    logger.info(
        f"""Nodes: '[{", ".join(nodes)}]' on data model '{data_model}' have incompatibility on the following cdes: '[{", ".join([cdes.__str__() for cdes in conflicting_cdes])}]' """
    )


def _log_duplicated_dataset(nodes, data_model, dataset):
    logger.info(
        f"""Dataset '{dataset}' of the data_model: '{data_model}' is not unique in the federation. Nodes that contain the dataset: '[{", ".join(nodes)}]'"""
    )
