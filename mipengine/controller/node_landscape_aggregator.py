import asyncio
import json
import sys
from typing import Dict
from typing import List
from typing import Tuple

import dns.resolver
from asgiref.sync import sync_to_async

from mipengine.controller import DeploymentType
from mipengine.controller import config as controller_config
from mipengine.controller import controller_logger as ctrl_logger
from mipengine.controller.celery_app import get_node_celery_app
from mipengine.controller.common_data_elements import CommonDataElement
from mipengine.controller.common_data_elements import CommonDataElements
from mipengine.controller.data_model_registry import data_model_registry
from mipengine.controller.node_registry import node_registry
from mipengine.node_exceptions import IncompatibleCDEs
from mipengine.node_info_DTOs import NodeInfo

NODE_LANDSCAPE_AGGREGATOR_REQUEST_ID = "NODE_LANDSCAPE_AGGREGATOR"
# TODO remove import get_node_celery_app, pass the celery app  (inverse dependency)
# so the module can be easily unit tested

logger = ctrl_logger.get_background_service_logger()

GET_NODE_INFO_SIGNATURE = "mipengine.node.tasks.common.get_node_info"
GET_NODE_DATASETS_PER_DATA_MODEL_SIGNATURE = (
    "mipengine.node.tasks.common.get_node_datasets_per_data_model"
)
GET_DATA_MODEL_CDES_SIGNATURE = "mipengine.node.tasks.common.get_data_model_cdes"
NODE_LANDSCAPE_AGGREGATOR_UPDATE_INTERVAL = (
    controller_config.node_registry_update_interval
)

CELERY_TASKS_TIMEOUT = controller_config.rabbitmq.celery_tasks_timeout


def _get_nodes_addresses_from_file() -> List[str]:
    with open(controller_config.localnodes.config_file) as fp:
        return json.load(fp)


def _get_nodes_addresses_from_dns() -> List[str]:
    localnodes_ips = dns.resolver.query(controller_config.localnodes.dns, "A")
    localnodes_addresses = [
        f"{ip}:{controller_config.localnodes.port}" for ip in localnodes_ips
    ]
    return localnodes_addresses


def _get_nodes_addresses() -> List[str]:
    if controller_config.deployment_type == DeploymentType.LOCAL:
        return _get_nodes_addresses_from_file()
    elif controller_config.deployment_type == DeploymentType.KUBERNETES:
        return _get_nodes_addresses_from_dns()
    else:
        return []


async def _get_nodes_info(nodes_socket_addr) -> List[NodeInfo]:
    celery_apps = [
        get_node_celery_app(socket_addr) for socket_addr in nodes_socket_addr
    ]
    nodes_task_signature = {
        celery_app: celery_app.signature(GET_NODE_INFO_SIGNATURE)
        for celery_app in celery_apps
    }

    # when broker(rabbitmq) is down, if the existing broker connection is not passed in
    # apply_async (in _task_to_async::wrapper), celery (or anyway some internal celery
    # component) will try to create a new connection to the broker until the apply_async
    # succeeds, which causes the call to apply_async to hang indefinitely until the
    # broker is back up. This way(passing the existing broker connection to apply_async)
    # it raises a ConnectionResetError or an OperationalError and it does not hang
    tasks_coroutines = [
        _task_to_async(task, connection=app.broker_connection())(
            request_id=NODE_LANDSCAPE_AGGREGATOR_REQUEST_ID
        )
        for app, task in nodes_task_signature.items()
    ]
    results = await asyncio.gather(*tasks_coroutines, return_exceptions=True)
    nodes_info = [
        NodeInfo.parse_raw(result)
        for result in results
        if not isinstance(result, Exception)
    ]
    return nodes_info


async def _get_node_datasets_per_data_model(
    node_socket_addr,
) -> Dict[str, Dict[str, str]]:
    celery_app = get_node_celery_app(node_socket_addr)
    task_signature = celery_app.signature(GET_NODE_DATASETS_PER_DATA_MODEL_SIGNATURE)

    # when broker(rabbitmq) is down, if the existing broker connection is not passed in
    # apply_async (in _task_to_async::wrapper), celery (or anyway some internal celery
    # component) will try to create a new connection to the broker until the apply_async
    # succeeds, which causes the call to apply_async to hang indefinitely until the
    # broker is back up. This way(passing the existing broker connection to apply_async)
    # it raises a ConnectionResetError or an OperationalError and it does not hang
    result = await _task_to_async(
        task_signature, connection=celery_app.broker_connection()
    )(request_id=NODE_LANDSCAPE_AGGREGATOR_REQUEST_ID)

    datasets_per_data_model = {}
    if not isinstance(result, Exception):
        datasets_per_data_model = {
            data_model: datasets for data_model, datasets in result.items()
        }
    return datasets_per_data_model


async def _get_node_cdes(node_socket_addr, data_model) -> CommonDataElements:
    celery_app = get_node_celery_app(node_socket_addr)
    task_signature = celery_app.signature(GET_DATA_MODEL_CDES_SIGNATURE)

    # when broker(rabbitmq) is down, if the existing broker connection is not passed in
    # apply_async (in _task_to_async::wrapper), celery (or anyway some internal celery
    # component) will try to create a new connection to the broker until the apply_async
    # succeeds, which causes the call to apply_async to hang indefinitely until the
    # broker is back up. This way(passing the existing broker connection to apply_async)
    # it raises a ConnectionResetError or an OperationalError and it does not hang
    result = await _task_to_async(
        task_signature, connection=celery_app.broker_connection()
    )(data_model=data_model, request_id=NODE_LANDSCAPE_AGGREGATOR_REQUEST_ID)

    cdes_per_data_model = {}
    if not isinstance(result, Exception):
        cdes_per_data_model = {
            code: CommonDataElement.parse_raw(metadata)
            for code, metadata in result.items()
        }
    cdes = CommonDataElements(cdes=cdes_per_data_model)
    return cdes


# Converts a Celery task to an async function
# Celery doesn't currently support asyncio "await" while "getting" a result
# Copied from https://github.com/celery/celery/issues/6603
def _task_to_async(task, connection):
    async def wrapper(*args, **kwargs):
        total_delay = 0
        delay = 0.1
        # Since apply_async is used instead of delay so that we can pass the connection as an argument,
        # the args and kwargs need to be passed as named arguments.
        async_result = await sync_to_async(task.apply_async)(
            args=args, kwargs=kwargs, connection=connection
        )
        while not async_result.ready():
            total_delay += delay
            if total_delay > CELERY_TASKS_TIMEOUT:
                raise TimeoutError(
                    f"Celery task: {task} didn't respond in {CELERY_TASKS_TIMEOUT}s."
                )
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 2)  # exponential backoff, max 2 seconds
        return async_result.get(timeout=CELERY_TASKS_TIMEOUT - total_delay)

    return wrapper


def _get_node_socket_addr(node_info: NodeInfo):
    return f"{node_info.ip}:{node_info.port}"


class NodeLandscapeAggregator:
    def __init__(self):
        self.keep_updating = True

    async def update(self):
        while self.keep_updating:
            nodes_addresses = _get_nodes_addresses()
            nodes: List[NodeInfo] = await _get_nodes_info(nodes_addresses)
            local_nodes = [node for node in nodes if node.role == "LOCALNODE"]
            datasets_locations = await _get_datasets_locations(local_nodes)
            datasets_labels = await _get_datasets_labels(local_nodes)
            node_cdes = await _get_cdes_across_nodes(local_nodes)
            common_data_models = _get_common_data_models(node_cdes)
            common_data_models = _get_common_data_models_with_valid_datasets(
                common_data_models, datasets_labels
            )
            # We are keeping only the data models that are common across nodes.
            datasets_locations = {
                common_data_model: datasets_locations[common_data_model]
                for common_data_model in common_data_models
            }

            node_registry.set_nodes(nodes)
            data_model_registry.set_common_data_models(common_data_models)
            data_model_registry.set_datasets_location(datasets_locations)
            logger.debug(f"Nodes:{[node.id for node in node_registry.nodes]}")
            # ..to print full nodes info
            # from devtools import debug
            # debug(self.nodes)
            # DEBUG end

            sys.stdout.flush()
            await asyncio.sleep(NODE_LANDSCAPE_AGGREGATOR_UPDATE_INTERVAL)


async def _get_datasets_locations(nodes):
    datasets_locations = {}
    for node_info in nodes:
        node_socket_addr = _get_node_socket_addr(node_info)
        datasets_per_data_model = await _get_node_datasets_per_data_model(
            node_socket_addr
        )
        for data_model, datasets in datasets_per_data_model.items():
            if data_model not in datasets_locations:
                datasets_locations[data_model] = {dataset: [] for dataset in datasets}
            for dataset in datasets:
                datasets_locations[data_model][dataset].append(node_info.id)
    return datasets_locations


async def _get_datasets_labels(nodes):
    datasets_labels = {}
    for node_info in nodes:
        node_socket_addr = _get_node_socket_addr(node_info)
        datasets_per_data_model = await _get_node_datasets_per_data_model(
            node_socket_addr
        )
        for data_model, datasets in datasets_per_data_model.items():
            datasets_labels[data_model] = {}
            for dataset in datasets:
                datasets_labels[data_model][dataset] = datasets[dataset]
    return datasets_labels


async def _get_cdes_across_nodes(nodes):
    nodes_cdes = {}
    for node_info in nodes:
        node_socket_addr = _get_node_socket_addr(node_info)
        datasets_per_data_model = await _get_node_datasets_per_data_model(
            node_socket_addr
        )
        for data_model in datasets_per_data_model:
            cdes = await _get_node_cdes(node_socket_addr, data_model)
            if data_model not in nodes_cdes:
                nodes_cdes[data_model] = []
            nodes_cdes[data_model].append((node_info.id, cdes))
    return nodes_cdes


def _get_common_data_models(
    nodes_cdes: Dict[str, List[Tuple[str, CommonDataElements]]]
) -> Dict[str, CommonDataElements]:
    common_data_models = {}
    for data_model, cdes_from_all_nodes in nodes_cdes.items():
        first_node, first_cdes = cdes_from_all_nodes[0]
        for node, cdes in cdes_from_all_nodes[1:]:
            try:
                first_cdes == cdes
            except IncompatibleCDEs as exc:
                logger.info(
                    f"Node '{first_node}' and node '{node}' on data model '{data_model}' threw: {exc.message}"
                )
                break
        else:
            common_data_models[data_model] = first_cdes

    return common_data_models


def _get_common_data_models_with_valid_datasets(
    common_data_models: Dict[str, CommonDataElements],
    datasets_per_data_model_across_nodes,
):
    for data_model in common_data_models:
        dataset_cde = common_data_models[data_model].cdes["dataset"]
        new_dataset_cde = CommonDataElement(
            code=dataset_cde.code,
            label=dataset_cde.label,
            sql_type=dataset_cde.sql_type,
            is_categorical=dataset_cde.is_categorical,
            enumerations=datasets_per_data_model_across_nodes[data_model],
            min=dataset_cde.min,
            max=dataset_cde.max,
        )
        common_data_models[data_model].cdes["dataset"] = new_dataset_cde
    return common_data_models


node_landscape_aggregator = NodeLandscapeAggregator()
