from typing import List

from mipengine.node_info_DTOs import NodeInfo
from mipengine.node_info_DTOs import NodeRole


class NodeRegistry:
    def __init__(self):
        self.nodes = []

    def set_nodes(self, nodes):
        self.nodes = nodes

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

    def get_nodes_by_ids(self, ids: List[str]) -> List[NodeInfo]:
        return [node for node in self.nodes if node.id in ids]


node_registry = NodeRegistry()
