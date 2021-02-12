import importlib.resources as pkg_resources
from dataclasses import dataclass
from typing import Dict
from typing import List

from dataclasses_json import dataclass_json

from mipengine import resources
from mipengine.controller.common.utils import Singleton


@dataclass_json
@dataclass
class GlobalNode:
    nodeId: str
    rabbitmqURL: str
    monetdbURL: str


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
class NodeCatalogue(metaclass=Singleton):
    _nodes: Nodes
    _datasets: Dict[str, List[str]]
    _nodes_per_dataset: Dict[str, List[LocalNode]]

    def __init__(self):
        node_catalogue_content = pkg_resources.read_text(resources, 'node_catalogue.json')
        self._nodes: Nodes = Nodes.from_json(node_catalogue_content)

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

    def pathology_exists(self, pathology: str):
        return pathology in self._datasets.keys()

    def dataset_exists(self, pathology: str, dataset: str):
        return dataset in self._datasets[pathology]

    def get_nodes_with_datasets(self, datasets: List[str]):
        return [self._nodes_per_dataset[dataset] for dataset in datasets]


NodeCatalogue()
