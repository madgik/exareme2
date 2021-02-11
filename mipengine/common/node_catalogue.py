import importlib.resources as pkg_resources
import json
from dataclasses import dataclass
from typing import Dict, List, Any

from dataclasses_json import dataclass_json

from mipengine import resources
from mipengine.controller.common.utils import Singleton


@dataclass_json
@dataclass
class Node:
    nodeId: str
    rabbitmqURL: str
    monetdbURL: str
    data: Dict[str, List[str]]


@dataclass_json
@dataclass
class NodeCatalogue(metaclass=Singleton):
    global_node: Node
    local_nodes: Dict[str, Node]
    data: Dict[str, List[str]]

    def __init__(self):
        node_catalogue_str = pkg_resources.read_text(resources, 'node_catalogue.json')
        node_catalogue: Dict[str, Any] = json.loads(node_catalogue_str)
        print(node_catalogue)
        self.global_node = node_catalogue["globalNode"]
        self.local_nodes = {(Node.from_dict(local_node)).nodeId: Node.from_dict(local_node)
                            for local_node in node_catalogue["localNodes"]}

        self.data = {}
        for local_node in self.local_nodes.values():
            for pathology_name, datasets in local_node.data.items():
                if pathology_name not in self.data.keys():
                    self.data[pathology_name] = datasets
                else:
                    self.data[pathology_name].extend(datasets)


NodeCatalogue()
