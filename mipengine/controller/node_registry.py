import asyncio
import json
import sys
from typing import Any
from typing import List

from celery import Celery
import dns.resolver

from mipengine.controller import DeploymentType
from mipengine.controller import config as controller_config
from mipengine.controller.celery_app import get_node_celery_app
from mipengine.node_info_DTOs import NodeInfo
from mipengine.node_info_DTOs import NodeRole

GET_NODE_INFO_SIGNATURE = "mipengine.node.tasks.common.get_node_info"
NODE_REGISTRY_UPDATE_INTERVAL = controller_config.node_registry_update_interval
NODE_REGISTRY_GET_NODE_INFO_TIMEOUT = (
    controller_config.node_registry_get_node_info_timeout
)


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


def _get_nodes_info(nodes_socket_addr) -> List[NodeInfo]:
    cel_apps = [get_node_celery_app(socket_addr) for socket_addr in nodes_socket_addr]
    nodes_task_signature = [app.signature(GET_NODE_INFO_SIGNATURE) for app in cel_apps]

    tasks_promise = []
    for task_signature in nodes_task_signature:
        try:
            tasks_promise.append(task_signature.delay())
        except:
            continue

    nodes_info = []
    for promise in tasks_promise:
        try:
            node_info = NodeInfo.parse_raw(
                promise.get(timeout=NODE_REGISTRY_GET_NODE_INFO_TIMEOUT)
            )
            nodes_info.append(node_info)
        except:
            continue
    return nodes_info


def _have_common_elements(a: List[Any], b: List[Any]):
    a_set = set(a)
    b_set = set(b)
    if len(a_set.intersection(b_set)) > 0:
        return True
    return False


class NodeRegistry:
    def __init__(self):
        self.nodes = []

    async def update(self):
        while True:
            nodes_addresses = _get_nodes_addresses()
            self.nodes: List[NodeInfo] = _get_nodes_info(nodes_addresses)
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
