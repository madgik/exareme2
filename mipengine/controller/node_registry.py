from typing import Dict

from mipengine.node_info_DTOs import NodeInfo
from mipengine.node_info_DTOs import NodeRole


class NodeRegistry:
    def __init__(self):
        self.nodes: Dict[str, NodeInfo] = {}

    def set_nodes(self, nodes: Dict[str, NodeInfo]):
        self.nodes = nodes

    def get_all_global_nodes(self) -> Dict[str, NodeInfo]:
        return {
            node_id: node_info
            for node_id, node_info in self.nodes.items()
            if node_info.role == NodeRole.GLOBALNODE
        }

    def get_all_local_nodes(self) -> Dict[str, NodeInfo]:
        return {
            node_id: node_info
            for node_id, node_info in self.nodes.items()
            if node_info.role == NodeRole.LOCALNODE
        }

    def get_node_info(self, node_id: str) -> NodeInfo:
        return self.nodes[node_id]


node_registry = NodeRegistry()
