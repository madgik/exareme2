from typing import Dict

from pydantic import BaseModel

from mipengine.node_info_DTOs import NodeInfo
from mipengine.node_info_DTOs import NodeRole


class NodeRegistry(BaseModel):
    nodes: Dict[str, NodeInfo] = {}

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True

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
