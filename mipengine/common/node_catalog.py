import importlib.resources as pkg_resources
from dataclasses import dataclass
from typing import Dict
from typing import List

from dataclasses_json import dataclass_json

from mipengine.common import resources
from mipengine.common.node_exceptions import InvalidNodeId


@dataclass_json
@dataclass
class GlobalNode:
    nodeId: str
    rabbitmqURL: str
    monetdbHostname: str
    monetdbPort: str


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
class LocalNode(GlobalNode):
    data: NodeData


@dataclass_json
@dataclass
class Nodes:
    globalNode: GlobalNode
    localNodes: List[LocalNode]


@dataclass_json
@dataclass
class NodeCatalog:
    _nodes: Nodes
    _datasets: Dict[str, List[str]]
    _nodes_per_dataset: Dict[str, List[LocalNode]]

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
                    else:
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

    def get_global_node(self) -> GlobalNode:
        return self._nodes.globalNode

    def get_local_nodes(self) -> List[LocalNode]:
        return self._nodes.localNodes

    def get_local_node(self, node_id) -> LocalNode:
        return next(
            local_node
            for local_node in self._nodes.localNodes
            if local_node.nodeId == node_id
        )


node_catalog = NodeCatalog()
