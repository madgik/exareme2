import pytest

from mipengine.controller.node_landscape_aggregator import DataModelRegistry
from mipengine.controller.node_landscape_aggregator import NodeLandscapeAggregator
from mipengine.controller.node_landscape_aggregator import NodeRegistry
from mipengine.node_info_DTOs import NodeInfo
from mipengine.node_info_DTOs import NodeRole

mocked_node_addresses = [
    "127.0.0.1:5672",
    "127.0.0.2:5672",
    "127.0.0.3:5672",
    "127.0.0.4:5672",
]


@pytest.fixture
def mocked_nla():
    nla = NodeLandscapeAggregator()
    nla.stop()
    nla.keep_updating = False

    node_registry = NodeRegistry(
        nodes_info=[
            NodeInfo(
                id="globalnode",
                role=NodeRole.GLOBALNODE,
                ip=mocked_node_addresses[0].split(":")[0],
                port=mocked_node_addresses[0].split(":")[1],
                db_ip="127.0.0.1",
                db_port=50000,
            ),
            NodeInfo(
                id="localnode1",
                role=NodeRole.LOCALNODE,
                ip=mocked_node_addresses[1].split(":")[0],
                port=mocked_node_addresses[1].split(":")[1],
                db_ip="127.0.0.1",
                db_port=50001,
            ),
            NodeInfo(
                id="localnode2",
                role=NodeRole.LOCALNODE,
                ip=mocked_node_addresses[2].split(":")[0],
                port=mocked_node_addresses[2].split(":")[1],
                db_ip="127.0.0.1",
                db_port=50002,
            ),
            NodeInfo(
                id="localnode3",
                role=NodeRole.LOCALNODE,
                ip=mocked_node_addresses[2].split(":")[0],
                port=mocked_node_addresses[2].split(":")[1],
                db_ip="127.0.0.1",
                db_port=50003,
            ),
        ]
    )
    nla._set_new_registries(
        node_registry=node_registry, data_model_registry=DataModelRegistry()
    )
    return nla


def test_get_nodes(mocked_nla):
    nodes = mocked_nla.get_nodes()
    assert len(nodes) == 4
    assert len([node for node in nodes if node.role == NodeRole.LOCALNODE]) == 3
    assert len([node for node in nodes if node.role == NodeRole.GLOBALNODE]) == 1


def test_get_global_node(mocked_nla):
    global_node = mocked_nla.get_global_node()
    assert global_node.role == NodeRole.GLOBALNODE


def test_get_all_local_nodes(mocked_nla):
    local_nodes = mocked_nla.get_all_local_nodes()
    assert len(local_nodes) == 3
    for node_info in local_nodes:
        assert node_info.role == NodeRole.LOCALNODE


def test_get_node_info(mocked_nla):
    expected_id = "localnode1"
    node_info = mocked_nla.get_node_info(expected_id)
    assert node_info.id == expected_id
    assert node_info.role == NodeRole.LOCALNODE
    assert node_info.db_port == 50001


def test_empty_initialization():
    node_registry = NodeRegistry()
    assert not node_registry.nodes_per_id.values()
