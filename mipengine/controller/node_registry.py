from typing import List

from pydantic import BaseModel

from mipengine.node_info_DTOs import NodeInfo
from mipengine.node_info_DTOs import NodeRole


class NodeRegistry(BaseModel):
    global_node: NodeInfo = None
    local_nodes: List[NodeInfo] = []

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True

    def get_nodes(self) -> List[NodeInfo]:
        return (
            self.local_nodes + [self.global_node]
            if self.global_node and self.local_nodes
            else []
        )

    def get_node_info(self, node_id: str) -> NodeInfo:
        return {node_info.id: node_info for node_info in self.get_nodes()}[node_id]
