import pytest
from mipengine.node_registry.node_registry import NodeRegistryClient
from mipengine.common.node_registry_DTOs import NodeRecord, Pathology, NodeRole
from ipaddress import IPv4Address

# A consul container will be started at the following port just for these tests
# After the tests finish the consul container is removed
CONSUL_TEST_PORT = 9500
CONSUL_TEST_CONTAINER_NAME = "pytest-consul"


@pytest.fixture(scope="session", autouse=True)
def start_test_consul_instance(request):

    import subprocess

    # start the consul container
    cmd = f"docker run -d --name={CONSUL_TEST_CONTAINER_NAME}  -p {CONSUL_TEST_PORT}:8500 consul"
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)

    import time

    timeout = time.time() + 10  # 10secs timeout
    is_container_running = False
    # check when container is in running state
    while not is_container_running:
        try:
            cmd = (
                "docker container inspect -f '{{.State.Running}}' "
                + CONSUL_TEST_CONTAINER_NAME
            )
            output = subprocess.check_output(cmd, shell=True)
            is_container_running = True if "true" in str(output) else False
        except subprocess.CalledProcessError:
            print("pytest-consul container is not RUNNING yet, retrying...")
            pass

        time.sleep(0.5)
        if time.time() > timeout:
            break

    # remove the consul container
    def teardown_container():
        cmd = f"docker rm -f {CONSUL_TEST_CONTAINER_NAME}"
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)

    request.addfinalizer(teardown_container)


@pytest.fixture
def node_registry_client():
    nrclient = NodeRegistryClient(
        consul_server_ip="127.0.0.1", consul_server_port=CONSUL_TEST_PORT
    )
    return nrclient


@pytest.fixture(autouse=True)
def test_register_nodes(node_registry_client, node_records):
    node_registry_client.register_node(node_records[0])
    node_registry_client.register_node(node_records[1])
    node_registry_client.register_node(node_records[2])


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


def test_get_registered_nodes(node_registry_client, expected_all_nodes_output):
    all_nodes = node_registry_client.get_all_nodes()
    assert all_nodes == expected_all_nodes_output


def test_get_global_nodes(node_registry_client, expected_global_nodes_output):
    global_nodes = node_registry_client.get_all_global_nodes()
    assert global_nodes == expected_global_nodes_output


def test_get_local_nodes(node_registry_client, expected_local_nodes_output):
    local_nodes = node_registry_client.get_all_local_nodes()
    assert local_nodes == expected_local_nodes_output


def test_get_nodes_with_pathologies(
    node_registry_client, expected_nodes_with_pathology2_and_pathology3
):
    nodes_with_pathologies = node_registry_client.get_nodes_with_pathologies(
        ["pathology2", "pathology3"]
    )
    assert nodes_with_pathologies == expected_nodes_with_pathology2_and_pathology3


def test_get_nodes_with_datasets(
    node_registry_client, expected_nodes_with_dataset2_and_dataset3
):
    nodes_with_datasets = node_registry_client.get_nodes_with_datasets(
        ["dataset2", "dataset3"]
    )
    assert nodes_with_datasets == expected_nodes_with_dataset2_and_dataset3


def test_get_db_info_for_node1(node_registry_client, expected_db_info_for_node1):
    db_info_node1 = node_registry_client.get_db("node1")
    assert db_info_node1 == expected_db_info_for_node1


def test_get_dbs_info(node_registry_client, expected_dbs_info):
    dbs_info = node_registry_client.get_dbs(["node1", "node2", "node3"])
    assert dbs_info == expected_dbs_info


def test_deregister_node(node_registry_client):
    node_registry_client.deregister_node("node1")
    all_nodes = list(node_registry_client.get_all_nodes().keys())
    pytest.assume("node1" not in all_nodes)
    with pytest.raises(node_registry_client.NodeIDNotInKVStore):
        node_registry_client.get_db("node1")
