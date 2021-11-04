import asyncio
import json
import sys
from typing import Any
from typing import List, Dict

import dns.resolver

from mipengine.controller import DeploymentType
from mipengine.controller import config as controller_config
from mipengine.controller.celery_app import get_node_celery_app
from mipengine.controller.celery_app import task_to_async
from mipengine.node_info_DTOs import NodeInfo
from mipengine.node_info_DTOs import NodeRole

# TODO remove import get_node_celery_app, pass the celery app  (inverse dependency)
# so the module can be easily unit tested

GET_NODE_INFO_SIGNATURE = "mipengine.node.tasks.common.get_node_info"
NODE_REGISTRY_UPDATE_INTERVAL = controller_config.node_registry_update_interval


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
    cel_apps = [get_node_celery_app(socket_addr) for socket_addr in nodes_socket_addr]
    nodes_task_signature = [app.signature(GET_NODE_INFO_SIGNATURE) for app in cel_apps]

    tasks_coroutines = [task_to_async(task)() for task in nodes_task_signature]
    results = await asyncio.gather(*tasks_coroutines, return_exceptions=True)
    nodes_info = [
        NodeInfo.parse_raw(result)
        for result in results
        if not isinstance(result, Exception)
    ]
    return nodes_info


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

            # DEBUG
            print(
                f"--> NodeRegistry just updated. Nodes:{[node.id for node in self.nodes]}"
            )
            # ..to print full nodes info
            # from devtools import debug
            # debug(self.nodes)
            # DEBUG end

            sys.stdout.flush()
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
