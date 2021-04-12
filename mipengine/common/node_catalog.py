import importlib.resources as pkg_resources
from dataclasses import dataclass
from typing import Dict
from typing import List

from dataclasses_json import dataclass_json

from mipengine.common import resources
from mipengine.common.node_exceptions import InvalidNodeId


@dataclass_json
@dataclass
class NodePathology:
    name: str
    datasets: List[str]


@dataclass_json
@dataclass
class NodeData:
    pathologies: List[NodePathology]


@dataclass_json
@dataclass
class Node:
    nodeId: str
    rabbitmqURL: str
    monetdbHostname: str
    monetdbPort: str


@dataclass_json
@dataclass
class GlobalNode(Node):
    pass


@dataclass_json
@dataclass
class LocalNode(Node):
    data: NodeData


@dataclass_json
@dataclass
class Nodes:
    globalNode: GlobalNode
    localNodes: List[LocalNode]


class NodeCatalog:
    def __init__(self):
        node_catalog_content = pkg_resources.read_text(resources, "node_catalog.json")
        self._nodes: Nodes = Nodes.from_json(node_catalog_content)

        for local_node in self._nodes.localNodes:
            if not local_node.nodeId.isalnum():
                raise InvalidNodeId(local_node.nodeId)
            if not local_node.nodeId.islower():
                raise ValueError(
                    f"Node id should be lower case, node id = {local_node.nodeId}"
                )

        self._datasets = {}
        for local_node in self._nodes.localNodes:
            for pathology in local_node.data.pathologies:
                if pathology.name not in self._datasets.keys():
                    self._datasets[pathology.name] = pathology.datasets
                else:
                    self._datasets[pathology.name].extend(pathology.datasets)

        self._nodes_per_dataset = {}
        for local_node in self._nodes.localNodes:
            for pathology in local_node.data.pathologies:
                for dataset in pathology.datasets:
                    if dataset not in self._nodes_per_dataset.keys():
                        self._nodes_per_dataset[dataset] = [local_node]
                    elif local_node not in self._nodes_per_dataset[dataset]:
                        self._nodes_per_dataset[dataset].append(local_node)

    def pathology_exists(self, pathology: str) -> bool:
        return pathology in self._datasets.keys()

    def dataset_exists(self, pathology: str, dataset: str) -> bool:
        return dataset in self._datasets[pathology]

    def get_nodes_with_datasets(self, datasets: List[str]) -> List[List[LocalNode]]:
        """
        Returns a List containing the LocalNodes ( List[LocalNode] ) that each
        dataset exists in.
        """
        return [self._nodes_per_dataset[dataset] for dataset in datasets]

    def get_nodes_with_any_of_datasets(self, datasets: List[str]) -> List[LocalNode]:
        tmp = self.get_nodes_with_datasets(datasets)
        if tmp:
            local_nodes = [tmp[0][0]]
            for nodes in tmp:
                for node in nodes:
                    if node not in local_nodes:
                        local_nodes.append(node)
            return local_nodes

        else:
            raise Exception(f"There are no nodes with any of the datasets->{datasets}")

    def get_global_node(self) -> GlobalNode:
        return self._nodes.globalNode

    def get_local_nodes(self) -> List[LocalNode]:
        return self._nodes.localNodes

    def get_local_node(self, node_id: str) -> LocalNode:
        try:
            return next(
                local_node
                for local_node in self._nodes.localNodes
                if local_node.nodeId == node_id
            )
        except StopIteration:
            raise ValueError(f"Node ID {node_id} not found in the Node Catalog")


node_catalog = NodeCatalog()
