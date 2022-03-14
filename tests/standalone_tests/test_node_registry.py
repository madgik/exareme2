from typing import List

import pytest

from mipengine.controller.node_registry import NodeRegistry
from mipengine.node_info_DTOs import NodeInfo
from mipengine.node_info_DTOs import NodeRole

mocked_node_addresses = [
    "127.0.0.1:5672",
    "127.0.0.2:5672",
    "127.0.0.3:5672",
    "127.0.0.4:5672",
]


def get_nodes_datasets_per_data_model():
    return {
        "globalnode": None,
        "localnode1": {
            "data_model1:0.1": [
                "dataset1",
                "dataset2",
                "dataset3",
                "dataset4",
                "dataset5",
            ],
            "data_model2:0.1": ["dataset6"],
        },
        "localnode2": {
            "data_model2:0.1": [
                "dataset7",
                "dataset8",
                "dataset9",
            ],
        },
        "localnode3": {
            "data_model2:0.1": [
                "dataset10",
            ],
        },
    }


def get_mocked_node_info() -> List[NodeInfo]:
    return [
        NodeInfo(
            id="globalnode",
            role=NodeRole.GLOBALNODE,
            ip=mocked_node_addresses[0].split(":")[0],
            port=mocked_node_addresses[0].split(":")[1],
            db_ip="127.0.0.1",
            db_port=50000,
            datasets_per_data_model=get_nodes_datasets_per_data_model()["globalnode"],
        ),
        NodeInfo(
            id="localnode1",
            role=NodeRole.LOCALNODE,
            ip=mocked_node_addresses[1].split(":")[0],
            port=mocked_node_addresses[1].split(":")[1],
            db_ip="127.0.0.1",
            db_port=50001,
            datasets_per_data_model=get_nodes_datasets_per_data_model()["localnode1"],
        ),
        NodeInfo(
            id="localnode2",
            role=NodeRole.LOCALNODE,
            ip=mocked_node_addresses[2].split(":")[0],
            port=mocked_node_addresses[2].split(":")[1],
            db_ip="127.0.0.1",
            db_port=50002,
            datasets_per_data_model=get_nodes_datasets_per_data_model()["localnode2"],
        ),
        NodeInfo(
            id="localnode3",
            role=NodeRole.LOCALNODE,
            ip=mocked_node_addresses[2].split(":")[0],
            port=mocked_node_addresses[2].split(":")[1],
            db_ip="127.0.0.1",
            db_port=50003,
            datasets_per_data_model=get_nodes_datasets_per_data_model()["localnode3"],
        ),
    ]


@pytest.fixture
def mocked_node_registry():
    node_registry = NodeRegistry()
    node_registry.nodes = get_mocked_node_info()
    return node_registry


def test_get_all_global_nodes(mocked_node_registry):
    global_nodes = mocked_node_registry.get_all_global_nodes()
    assert len(global_nodes) == 1
    assert global_nodes[0].role == NodeRole.GLOBALNODE


def test_get_all_local_nodes(mocked_node_registry):
    local_nodes = mocked_node_registry.get_all_local_nodes()
    assert len(local_nodes) == 3
    for node in local_nodes:
        assert node.role == NodeRole.LOCALNODE


def test_get_nodes_by_ids(mocked_node_registry):
    expected_ids = ["localnode1", "localnode3"]
    nodes = mocked_node_registry.get_nodes_by_ids(expected_ids)
    assert set([node.id for node in nodes]) == set(expected_ids)
