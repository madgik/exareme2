from typing import Dict

from mipengine.node_info_DTOs import NodeInfo
from mipengine.node_info_DTOs import NodeRole


class NodeRegistry:
    def __init__(self):
        self._nodes: Dict[str, NodeInfo] = {}

    @property
    def nodes(self) -> Dict[str, NodeInfo]:
        return self._nodes

    @nodes.setter
    def nodes(self, values):
        self._nodes = values

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
