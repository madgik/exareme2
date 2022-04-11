from logging import Logger
from typing import Dict

from mipengine.node_info_DTOs import NodeInfo
from mipengine.node_info_DTOs import NodeRole


class NodeRegistry:
    def __init__(self, logger: Logger):
        self._logger = logger
        self._nodes: Dict[str, NodeInfo] = {}

    @property
    def nodes(self) -> Dict[str, NodeInfo]:
        return self._nodes

    @nodes.setter
    def nodes(self, value):
        log_node_changes(self._logger, self._nodes, value)
        self._nodes = value

    def _get_all_global_nodes(self) -> Dict[str, NodeInfo]:
        return {
            node_id: node_info
            for node_id, node_info in self.nodes.items()
            if node_info.role == NodeRole.GLOBALNODE
        }

    def get_global_node(self) -> NodeInfo:
        return list(self._get_all_global_nodes().values())[0]

    def get_all_local_nodes(self) -> Dict[str, NodeInfo]:
        return {
            node_id: node_info
            for node_id, node_info in self.nodes.items()
            if node_info.role == NodeRole.LOCALNODE
        }

    def get_node_info(self, node_id: str) -> NodeInfo:
        return self.nodes[node_id]


def log_node_changes(logger, old_nodes, new_nodes):
    added_nodes = set(new_nodes.keys()) - set(old_nodes.keys())
    for node in added_nodes:
        logger.info(f"Node with id '{node}' joined the federation.")

    removed_nodes = set(old_nodes.keys()) - set(new_nodes.keys())
    for node in removed_nodes:
        logger.info(f"Node with id '{node}' left the federation.")
