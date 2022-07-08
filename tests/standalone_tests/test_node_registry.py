from typing import Dict

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


def get_mocked_node_info() -> Dict[str, NodeInfo]:
    return {
        "globalnode": NodeInfo(
            id="globalnode",
            role=NodeRole.GLOBALNODE,
            ip=mocked_node_addresses[0].split(":")[0],
            port=mocked_node_addresses[0].split(":")[1],
            db_ip="127.0.0.1",
            db_port=50000,
        ),
        "localnode1": NodeInfo(
            id="localnode1",
            role=NodeRole.LOCALNODE,
            ip=mocked_node_addresses[1].split(":")[0],
            port=mocked_node_addresses[1].split(":")[1],
            db_ip="127.0.0.1",
            db_port=50001,
        ),
        "localnode2": NodeInfo(
            id="localnode2",
            role=NodeRole.LOCALNODE,
            ip=mocked_node_addresses[2].split(":")[0],
            port=mocked_node_addresses[2].split(":")[1],
            db_ip="127.0.0.1",
            db_port=50002,
        ),
        "localnode3": NodeInfo(
            id="localnode3",
            role=NodeRole.LOCALNODE,
            ip=mocked_node_addresses[2].split(":")[0],
            port=mocked_node_addresses[2].split(":")[1],
            db_ip="127.0.0.1",
            db_port=50003,
        ),
    }


@pytest.fixture
def mocked_node_registry():
    node_registry = NodeRegistry(nodes=get_mocked_node_info())
    return node_registry


def test_get_global_node(mocked_node_registry):
    global_node = mocked_node_registry.get_global_node()
    assert global_node.role == NodeRole.GLOBALNODE


def test_get_all_local_nodes(mocked_node_registry):
    local_nodes = mocked_node_registry.get_all_local_nodes()
    assert len(local_nodes) == 3
    for node_info in local_nodes:
        assert local_nodes[node_info].role == NodeRole.LOCALNODE


def test_get_node_info(mocked_node_registry):
    expected_id = "localnode1"
    node_info = mocked_node_registry.get_node_info(expected_id)
    assert node_info.id == expected_id
    assert node_info.role == NodeRole.LOCALNODE
    assert node_info.db_port == 50001
