import asyncio
import json
import sys
from typing import Any
from typing import List, Dict

from celery import Celery
from asgiref.sync import sync_to_async

import dns.resolver

from mipengine.controller import DeploymentType
from mipengine.controller import config as controller_config
from mipengine.controller.celery_app import get_node_celery_app

from mipengine.node_info_DTOs import NodeInfo
from mipengine.node_info_DTOs import NodeRole
from mipengine.controller import controller_logger as ctrl_logger


logger = ctrl_logger.get_background_service_logger()

GET_NODE_INFO_SIGNATURE = "mipengine.node.tasks.common.get_node_info"
NODE_REGISTRY_UPDATE_INTERVAL = controller_config.node_registry_update_interval

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
    # succeeds, wich causes the call to apply_async to hang indefinetelly until the
    # broker is back up. This way(passing the existing broker connection to apply_async)
    # it raises a ConnectionResetError or an OperationalError and it does not hang
    tasks_coroutines = [
        _task_to_async(task)(connection=app.broker_connection())
        for app, task in nodes_task_signature.items()
    ]

    results = await asyncio.gather(*tasks_coroutines, return_exceptions=True)
    nodes_info = [
        NodeInfo.parse_raw(result)
        for result in results
        if not isinstance(result, Exception)
    ]

    return nodes_info


# Converts a Celery task to an async function
# Celery doesn't currently support asyncio "await" while "getting" a result
# Copied from https://github.com/celery/celery/issues/6603
def _task_to_async(task):
    async def wrapper(*args, **kwargs):
        total_delay = 0
        delay = 0.1
        async_result = await sync_to_async(task.apply_async)(*args, **kwargs)
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


def _have_common_elements(a: List[Any], b: List[Any]):
    return bool(set(a) & set(b))


class NodeRegistry:
    def __init__(self):
        self.nodes = []
        self.keep_updating = True

    async def update(self):
        while self.keep_updating:
            nodes_addresses = _get_nodes_addresses()

            self.nodes: List[NodeInfo] = await _get_nodes_info(nodes_addresses)

            logger.debug(f"Nodes:{[node.id for node in self.nodes]}")
            # ..to print full nodes info
            # from devtools import debug
            # debug(self.nodes)
            # DEBUG end

            sys.stdout.flush()  # TODO what is this for??
            await asyncio.sleep(NODE_REGISTRY_UPDATE_INTERVAL)

    def get_all_global_nodes(self) -> List[NodeInfo]:
        return [
            node_info
            for node_info in self.nodes
            if node_info.role == NodeRole.GLOBALNODE
        ]

    def get_all_local_nodes(self) -> List[NodeInfo]:
        return [
            node_info
            for node_info in self.nodes
            if node_info.role == NodeRole.LOCALNODE
        ]

    def get_nodes_with_any_of_datasets(
        self, schema: str, datasets: List[str]
    ) -> List[NodeInfo]:
        all_local_nodes = self.get_all_local_nodes()
        local_nodes_with_datasets = []
        for node_info in all_local_nodes:
            if not node_info.datasets_per_schema:
                continue
            if schema not in node_info.datasets_per_schema:
                continue
            if not _have_common_elements(
                node_info.datasets_per_schema[schema], datasets
            ):
                continue
            local_nodes_with_datasets.append(node_info)

        return local_nodes_with_datasets

    # returns a list of all the currently availiable schemas(patholgies) on the system
    # without duplicates
    def get_all_available_schemas(self) -> List[str]:
        all_local_nodes = self.get_all_local_nodes()
        tmp = [local_node.datasets_per_schema for local_node in all_local_nodes]
        all_existing_schemas = set().union(*tmp)
        return list(all_existing_schemas)

    # returns a dictionary with all the currently availiable schemas(patholgies) on the
    # system as keys and lists of datasets as values. Without duplicates
    def get_all_available_datasets_per_schema(self) -> Dict[str, List[str]]:
        all_local_nodes = self.get_all_local_nodes()
        tmp = [node_info.datasets_per_schema for node_info in all_local_nodes]

        from collections import defaultdict
        from itertools import chain
        from operator import methodcaller

        dd = defaultdict(list)
        dict_items = map(methodcaller("items"), tmp)
        for k, v in chain.from_iterable(dict_items):
            dd[k].extend(v)
        return dict(dd)

    def schema_exists(self, schema: str):
        for node_info in self.get_all_local_nodes():
            if not node_info.datasets_per_schema:
                continue
            if schema in node_info.datasets_per_schema.keys():
                return True
        return False

    def dataset_exists(self, schema: str, dataset: str):
        for node_info in self.get_all_local_nodes():
            if not node_info.datasets_per_schema:
                continue
            if schema not in node_info.datasets_per_schema:
                continue
            if not _have_common_elements(
                node_info.datasets_per_schema[schema], [dataset]
            ):
                continue
            return True
        return False


node_registry = NodeRegistry()
