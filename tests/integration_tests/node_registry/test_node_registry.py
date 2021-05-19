import pytest
from mipengine.node_registry.node_registry import NodeRegistryClient
from mipengine.common.node_registry_DTOs import NodeRecord, Pathology, NodeRole
from ipaddress import IPv4Address

# TESTS PREREQUISITES
# these tests expect consul agent to be running at 127.0.0.1:8500 and without any
# services registered
# To start consul: $./consul agent -dev or $docker run -d --name=dev-consul  -p 8500:8500 consul


@pytest.fixture
def node_records():
    a_pathology1 = Pathology(name="pathology1", datasets=["dataset1", "dataset2"])
    a_node_record_1 = NodeRecord(
        node_id="node1",
        node_role=NodeRole.GLOBALNODE,
        task_queue_ip=IPv4Address("127.0.0.123"),
        task_queue_port=1234,
        db_id="node1_db",
        db_ip=IPv4Address("127.0.0.1"),
        db_port=5678,
        pathologies=[a_pathology1],
    )
    a_pathology2 = Pathology(name="pathology2", datasets=["dataset3", "dataset4"])
    a_node_record_2 = NodeRecord(
        node_id="node2",
        node_role=NodeRole.LOCALNODE,
        task_queue_ip=IPv4Address("127.0.0.124"),
        task_queue_port=1234,
        db_id="node2_db",
        db_ip=IPv4Address("127.0.0.2"),
        db_port=5678,
        pathologies=[a_pathology1, a_pathology2],
    )

    a_pathology3 = Pathology(name="pathology3", datasets=["dataset5", "dataset6"])
    a_node_record_3 = NodeRecord(
        node_id="node3",
        node_role=NodeRole.LOCALNODE,
        task_queue_ip=IPv4Address("127.0.0.125"),
        task_queue_port=1234,
        db_id="node3_db",
        db_ip=IPv4Address("127.0.0.3"),
        db_port=5678,
        pathologies=[a_pathology2, a_pathology3],
    )
    return [a_node_record_1, a_node_record_2, a_node_record_3]


@pytest.fixture
def expected_all_nodes_output():
    nodes = {}
    nodes["node1"] = NodeRegistryClient.NodeInfo(
        ip=IPv4Address("127.0.0.123"),
        port=1234,
        role=NodeRole.GLOBALNODE,
        pathologies=[Pathology(name="pathology1", datasets=["dataset1", "dataset2"])],
    )
    nodes["node2"] = NodeRegistryClient.NodeInfo(
        ip=IPv4Address("127.0.0.124"),
        port=1234,
        role=NodeRole.LOCALNODE,
        pathologies=[
            Pathology(name="pathology1", datasets=["dataset1", "dataset2"]),
            Pathology(name="pathology2", datasets=["dataset3", "dataset4"]),
        ],
    )
    nodes["node3"] = NodeRegistryClient.NodeInfo(
        ip=IPv4Address("127.0.0.125"),
        port=1234,
        role=NodeRole.LOCALNODE,
        pathologies=[
            Pathology(name="pathology2", datasets=["dataset3", "dataset4"]),
            Pathology(name="pathology3", datasets=["dataset5", "dataset6"]),
        ],
    )
    return nodes


@pytest.fixture
def expected_global_nodes_output():
    nodes = {}
    nodes["node1"] = NodeRegistryClient.NodeInfo(
        ip=IPv4Address("127.0.0.123"),
        port=1234,
        role=NodeRole.GLOBALNODE,
        pathologies=[Pathology(name="pathology1", datasets=["dataset1", "dataset2"])],
    )
    return nodes


@pytest.fixture
def expected_local_nodes_output():
    nodes = {}
    nodes["node2"] = NodeRegistryClient.NodeInfo(
        ip=IPv4Address("127.0.0.124"),
        port=1234,
        role=NodeRole.LOCALNODE,
        pathologies=[
            Pathology(name="pathology1", datasets=["dataset1", "dataset2"]),
            Pathology(name="pathology2", datasets=["dataset3", "dataset4"]),
        ],
    )
    nodes["node3"] = NodeRegistryClient.NodeInfo(
        ip=IPv4Address("127.0.0.125"),
        port=1234,
        role=NodeRole.LOCALNODE,
        pathologies=[
            Pathology(name="pathology2", datasets=["dataset3", "dataset4"]),
            Pathology(name="pathology3", datasets=["dataset5", "dataset6"]),
        ],
    )
    return nodes


@pytest.fixture
def expected_nodes_with_pathology2_and_pathology3():
    nodes = {}
    nodes["node3"] = NodeRegistryClient.NodeInfo(
        ip=IPv4Address("127.0.0.125"),
        port=1234,
        role=NodeRole.LOCALNODE,
        pathologies=[
            Pathology(name="pathology2", datasets=["dataset3", "dataset4"]),
            Pathology(name="pathology3", datasets=["dataset5", "dataset6"]),
        ],
    )
    return nodes


@pytest.fixture
def expected_nodes_with_dataset2_and_dataset3():
    nodes = {}
    nodes["node2"] = NodeRegistryClient.NodeInfo(
        ip=IPv4Address("127.0.0.124"),
        port=1234,
        role=NodeRole.LOCALNODE,
        pathologies=[
            Pathology(name="pathology1", datasets=["dataset1", "dataset2"]),
            Pathology(name="pathology2", datasets=["dataset3", "dataset4"]),
        ],
    )
    return nodes


@pytest.fixture
def expected_db_info_for_node1():
    db_info_node1 = NodeRegistryClient.DBInfo(
        id="node1_db", ip=IPv4Address("127.0.0.1"), port=5678
    )
    return db_info_node1


@pytest.fixture
def expected_dbs_info():
    dbs_info = {}
    dbs_info["node1"] = NodeRegistryClient.DBInfo(
        id="node1_db", ip=IPv4Address("127.0.0.1"), port=5678
    )
    dbs_info["node2"] = NodeRegistryClient.DBInfo(
        id="node2_db", ip=IPv4Address("127.0.0.2"), port=5678
    )
    dbs_info["node3"] = NodeRegistryClient.DBInfo(
        id="node3_db", ip=IPv4Address("127.0.0.3"), port=5678
    )
    return dbs_info


@pytest.fixture
def node_registry_client():
    # nrclient = NodeRegistryClient(consul_server_ip="127.0.0.1", consul_server_port=8500)
    nrclient = NodeRegistryClient()
    return nrclient


def test_register_nodes(node_registry_client, node_records):
    node_registry_client.register_node(node_records[0])
    node_registry_client.register_node(node_records[1])
    node_registry_client.register_node(node_records[2])
    assert True


def test_get_registered_nodes(node_registry_client, expected_all_nodes_output):
    all_nodes = node_registry_client.get_all_nodes()
    print("NodeRegistryClient returned:")
    {print(f"{node_id=}\n{node_info=}") for (node_id, node_info) in all_nodes.items()}
    print("\n DO NOT MATCH EXPECTED-->\n")
    print("expected_registered_nodes_output:")
    {
        print(f"{node_id=}\n{node_info=}")
        for (node_id, node_info) in expected_all_nodes_output.items()
    }
    assert all_nodes == expected_all_nodes_output


def test_get_global_nodes(node_registry_client, expected_global_nodes_output):
    global_nodes = node_registry_client.get_all_global_nodes()
    print("NodeRegistryClient returned:")
    {
        print(f"{node_id=}\n{node_info=}")
        for (node_id, node_info) in global_nodes.items()
    }
    print("\n DO NOT MATCH EXPECTED-->\n")
    print("expected_global_nodes_output:")
    {
        print(f"{node_id=}\n{node_info=}")
        for (node_id, node_info) in expected_global_nodes_output.items()
    }

    assert global_nodes == expected_global_nodes_output


def test_get_local_nodes(node_registry_client, expected_local_nodes_output):
    local_nodes = node_registry_client.get_all_local_nodes()
    print("NodeRegistryClient returned:")
    {print(f"{node_id=}\n{node_info=}") for (node_id, node_info) in local_nodes.items()}
    print("\n DO NOT MATCH EXPECTED-->\n")
    print("expected_local_nodes_output:")
    {
        print(f"{node_id=}\n{node_info=}")
        for (node_id, node_info) in expected_local_nodes_output.items()
    }

    assert local_nodes == expected_local_nodes_output


def test_get_nodes_with_pathologies(
    node_registry_client, expected_nodes_with_pathology2_and_pathology3
):
    nodes_with_pathologies = node_registry_client.get_nodes_with_pathologies(
        ["pathology2", "pathology3"]
    )
    print("NodeRegistryClient returned:")
    {
        print(f"{node_id=}\n{node_info=}")
        for (node_id, node_info) in nodes_with_pathologies.items()
    }
    print("\n DO NOT MATCH EXPECTED-->\n")
    print("expected_nodes_with_pathology2_and_pathology3:")
    {
        print(f"{node_id=}\n{node_info=}")
        for (
            node_id,
            node_info,
        ) in expected_nodes_with_pathology2_and_pathology3.items()
    }

    assert nodes_with_pathologies == expected_nodes_with_pathology2_and_pathology3


def test_get_nodes_with_datasets(
    node_registry_client, expected_nodes_with_dataset2_and_dataset3
):
    nodes_with_datasets = node_registry_client.get_nodes_with_datasets(
        ["dataset2", "dataset3"]
    )
    print("NodeRegistryClient returned:")
    {
        print(f"{node_id=}\n{node_info=}")
        for (node_id, node_info) in nodes_with_datasets.items()
    }
    print("\n DO NOT MATCH EXPECTED-->\n")
    print("expected_nodes_with_dataset2_and_dataset3:")
    {
        print(f"{node_id=}\n{node_info=}")
        for (node_id, node_info) in expected_nodes_with_dataset2_and_dataset3.items()
    }

    assert nodes_with_datasets == expected_nodes_with_dataset2_and_dataset3


def test_get_db_info_for_node1(node_registry_client, expected_db_info_for_node1):
    db_info_node1 = node_registry_client.get_db("node1")
    print("NodeRegistryClient returned:")
    print(f"{db_info_node1=}")
    print("\n DO NOT MATCH EXPECTED-->\n")
    print("expected_db_info_for_node1:")
    print(f"{expected_db_info_for_node1}")

    assert db_info_node1 == expected_db_info_for_node1


def test_get_dbs_info(node_registry_client, expected_dbs_info):
    dbs_info = node_registry_client.get_dbs(["node1", "node2", "node3"])
    print("NodeRegistryClient returned:")
    {print(f"{node_id=}\n{db_info=}") for (node_id, db_info) in dbs_info.items()}
    print("\n DO NOT MATCH EXPECTED-->\n")
    print("expected_all_dbs_info:")
    {
        print(f"{node_id=}\n{node_info=}")
        for (node_id, node_info) in expected_dbs_info.items()
    }

    assert dbs_info == expected_dbs_info


def test_deregister_node(node_registry_client):
    node_registry_client.deregister_node("node1")
    all_nodes = list(node_registry_client.get_all_nodes().keys())
    pytest.assume("node1" not in all_nodes)
    with pytest.raises(node_registry_client.NodeIDNotInKVStore):
        node_registry_client.get_db("node1")
