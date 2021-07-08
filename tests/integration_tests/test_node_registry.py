import pytest
from mipengine.node_registry import (
    NodeRegistryClient,
    NodeRole,
    Pathology,
    Pathologies,
    NodeParams,
    DBParams,
    DBParamsIdNotMatchingNodeDBId,
    NodeIDUnknown,
    DBIdUnknown,
    NodeIdNotFoundInKVStore,
    GlobalNodeCannotContainPrimaryData,
    LocalNodeMustContainPrimaryData,
)
from ipaddress import IPv4Address

import subprocess
import time

# A consul container will be started at the following port just for these tests
# After the tests finish the consul container is removed
CONSUL_TEST_PORT = 9500
CONSUL_TEST_CONTAINER_NAME = "pytest-consul"


@pytest.fixture(scope="session", autouse=True)
def start_test_consul_instance(request):

    # start the consul container
    cmd = f"docker run -d --name={CONSUL_TEST_CONTAINER_NAME}  -p {CONSUL_TEST_PORT}:8500 consul"
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)

    timeout = time.time() + 10  # 10secs timeout
    is_container_running = False
    # check when container is in running state,it takes a couple of seconds
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

    # remove the test consul container
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


@pytest.fixture
def test_nodes():
    globalnode1_params = NodeParams(
        id="globalnode1",
        ip=IPv4Address("127.0.0.123"),
        port=1234,
        role=NodeRole.GLOBALNODE,
        db_id="globalnode1_db",
    )
    localnode1_params = NodeParams(
        id="localnode1",
        role=NodeRole.LOCALNODE,
        ip=IPv4Address("127.0.0.124"),
        port=1235,
        db_id="localnode1_db",
    )

    localnode2_params = NodeParams(
        id="localnode2",
        role=NodeRole.LOCALNODE,
        ip=IPv4Address("127.0.0.125"),
        port=1236,
        db_id="localnode2_db",
    )
    localnode3_params = NodeParams(
        id="localnode3",
        role=NodeRole.LOCALNODE,
        ip=IPv4Address("127.0.0.126"),
        port=1237,
        db_id="localnode3_db",
    )

    return {
        "globalnode1": globalnode1_params,
        "localnode1": localnode1_params,
        "localnode2": localnode2_params,
        "localnode3": localnode3_params,
    }


@pytest.fixture
def test_dbs():
    pathology1 = Pathology(name="pathology1", datasets=["dataset1", "dataset2"])
    pathology2 = Pathology(name="pathology2", datasets=["dataset3", "dataset4"])
    pathology3 = Pathology(name="pathology3", datasets=["dataset5", "dataset6"])
    pathology4 = Pathology(name="pathology4", datasets=["dataset7", "dataset8"])

    dbs_params = {}

    dbs_params["globalnode1"] = DBParams(
        id="globalnode1_db", ip=IPv4Address("127.0.0.1"), port=5678
    )

    dbs_params["localnode1"] = DBParams(
        id="localnode1_db",
        ip=IPv4Address("127.0.0.2"),
        port=5679,
        pathologies=Pathologies(pathologies_list=[pathology1, pathology2]),
    )

    dbs_params["localnode2"] = DBParams(
        id="localnode2_db",
        ip=IPv4Address("127.0.0.3"),
        port=5680,
        pathologies=Pathologies(pathologies_list=[pathology2, pathology3]),
    )

    dbs_params["localnode3"] = DBParams(
        id="localnode3_db",
        ip=IPv4Address("127.0.0.4"),
        port=5681,
        pathologies=Pathologies(pathologies_list=[pathology3, pathology4]),
    )
    return dbs_params


@pytest.fixture(autouse=True)
def register_nodes(node_registry_client, test_nodes, test_dbs):
    node_registry_client.register_node(
        test_nodes["globalnode1"], test_dbs["globalnode1"]
    )
    node_registry_client.register_node(test_nodes["localnode1"], test_dbs["localnode1"])
    node_registry_client.register_node(test_nodes["localnode2"], test_dbs["localnode2"])
    node_registry_client.register_node(test_nodes["localnode3"], test_dbs["localnode3"])


@pytest.fixture
def expected_all_nodes(test_nodes):
    return list(test_nodes.values())


@pytest.fixture
def expected_global_node(test_nodes):
    return test_nodes["globalnode1"]


@pytest.fixture
def expected_local_nodes(test_nodes):
    return [
        test_nodes["localnode1"],
        test_nodes["localnode2"],
        test_nodes["localnode3"],
    ]


@pytest.fixture
def expected_nodes_with_pathology2_and_pathology3(test_nodes):
    return [test_nodes["localnode2"]]


@pytest.fixture
def expected_nodes_with_dataset2_or_dataset3(test_nodes):
    return [test_nodes["localnode1"], test_nodes["localnode2"]]


@pytest.fixture
def expected_all_dbs(test_dbs):
    return list(test_dbs.values())


@pytest.fixture
def expected_db_for_localnode1(test_dbs):
    return test_dbs["localnode1"]


@pytest.fixture
def expected_db_with_pathology2_and_pathology3(test_dbs):
    return [test_dbs["localnode2"]]


@pytest.fixture
def expected_db_with_dataset2_or_dataset3(test_dbs):
    return [test_dbs["localnode1"], test_dbs["localnode2"]]


def test_get_all_registered_nodes(node_registry_client, expected_all_nodes):
    all_nodes = node_registry_client.get_all_nodes()
    assert all_nodes == expected_all_nodes


def test_get_node_by_node_id(node_registry_client, test_nodes):
    for node_id, node_expected in test_nodes.items():
        node = node_registry_client.get_node_by_node_id(node_id)
        pytest.assume(node == node_expected)


def test_get_global_nodes(node_registry_client, expected_global_node):
    global_nodes = node_registry_client.get_all_global_nodes()
    assert global_nodes[0] == expected_global_node


def test_get_local_nodes(node_registry_client, expected_local_nodes):
    local_nodes = node_registry_client.get_all_local_nodes()
    assert local_nodes == expected_local_nodes


def test_get_pathologies_by_db_id(
    node_registry_client, expected_db_with_pathology2_and_pathology3
):
    # test that given a db_id the contained pathologies are returned
    for db_params in expected_db_with_pathology2_and_pathology3:
        db_id = db_params.id
        pathologies = node_registry_client.get_pathologies_by_db_id(db_id)
        expected_pathologies = db_params.pathologies
        assert pathologies == expected_pathologies


def test_get_datasets_by_db_id(
    node_registry_client, expected_db_with_dataset2_or_dataset3
):
    # test that given a db_id the contained datasets are returned
    for db_params in expected_db_with_dataset2_or_dataset3:
        datasets = node_registry_client.get_datasets_by_db_id(db_params.id)
        expected_datasets = [
            dataset
            for pathologies in db_params.pathologies.pathologies_list
            for dataset in pathologies.datasets
        ]
        assert _compare_unordered_lists(datasets, expected_datasets)


def test_get_nodes_with_pathologies(
    node_registry_client, expected_nodes_with_pathology2_and_pathology3
):
    nodes_with_pathologies = node_registry_client.get_nodes_with_all_of_pathologies(
        ["pathology2", "pathology3"]
    )
    assert _compare_unordered_lists(
        nodes_with_pathologies, expected_nodes_with_pathology2_and_pathology3
    )


def test_get_nodes_with_any_of_datasets(
    node_registry_client, expected_nodes_with_dataset2_or_dataset3
):
    nodes_with_datasets = node_registry_client.get_nodes_with_any_of_datasets(
        ["dataset2", "dataset3"]
    )
    assert nodes_with_datasets == expected_nodes_with_dataset2_or_dataset3


def test_get_db_params_for_node1(node_registry_client, expected_db_for_localnode1):
    db_params = node_registry_client.get_db_by_node_id("localnode1")
    assert db_params == expected_db_for_localnode1


def test_get_node_id_by_db_id(node_registry_client, expected_all_nodes):
    for expected_node in expected_all_nodes:
        node_id = node_registry_client.get_node_id_by_db_id(expected_node.db_id)
        assert node_id == expected_node.id


def test_get_all_dbs(node_registry_client, expected_all_dbs):
    all_dbs = node_registry_client.get_all_dbs()
    assert _compare_unordered_lists(all_dbs, expected_all_dbs)


def test_deregister_node(node_registry_client):
    node_id = "localnode1"
    node_registry_client.deregister_node(node_id)
    all_nodes = node_registry_client.get_all_nodes()
    pytest.assume(node_id not in all_nodes)

    with pytest.raises(NodeIdNotFoundInKVStore):
        node_registry_client.get_db_by_node_id(node_id)


def test_pathology_exists(node_registry_client):
    pytest.assume(node_registry_client.pathology_exists("pathology1"))
    pytest.assume(not node_registry_client.pathology_exists("non_existing_pathology"))


# def test_dataset_exists(node_registry_client):
#     pytest.assume(node_registry_client.dataset_exists("dataset1"))
#     pytest.assume(not node_registry_client.dataset_exists("non_existing_dataset"))
def test_dataset_exists(node_registry_client):
    pytest.assume(node_registry_client.dataset_exists("pathology1", "dataset1"))
    pytest.assume(
        not node_registry_client.dataset_exists("pathology1", "non_existing_dataset")
    )
    pytest.assume(
        not node_registry_client.dataset_exists("non_existing_pathology", "dataset1")
    )


# test for exceptions...
def test_get_node_unknown_node_id(node_registry_client):
    with pytest.raises(NodeIDUnknown):
        node_registry_client.get_node_by_node_id("this_is_a_non_existing_node_id")


def test_deregister_unknown_node(node_registry_client):
    with pytest.raises(NodeIDUnknown):
        node_registry_client.deregister_node("this_is_a_non_existing_node_id")


def test_get_db_unknown_node_id(node_registry_client):
    with pytest.raises(NodeIdNotFoundInKVStore):
        node_registry_client.get_db_by_node_id("this_is_a_non_existing_node_id")


def test_get_db_unknown_db_id(node_registry_client):
    with pytest.raises(DBIdUnknown):
        node_registry_client.get_db_by_db_id("this_is_a_non_existing_node_id")


def test_not_matching_db_ids_exception(node_registry_client, test_nodes, test_dbs):
    node = test_nodes["globalnode1"]
    db = test_dbs["globalnode1"]
    db.id = "changed_to_something_else"
    with pytest.raises(DBParamsIdNotMatchingNodeDBId):
        node_registry_client.register_node(node_params=node, db_params=db)


def test_global_node_with_primary_data_exception(
    node_registry_client, test_nodes, test_dbs
):
    node = test_nodes["localnode1"]
    node.role = NodeRole.GLOBALNODE
    db = test_dbs["localnode1"]
    with pytest.raises(GlobalNodeCannotContainPrimaryData):
        node_registry_client.register_node(node_params=node, db_params=db)


def test_local_node_without_primary_data_exception(
    node_registry_client, test_nodes, test_dbs
):
    node = test_nodes["localnode1"]
    db = test_dbs["localnode1"]
    db.pathologies = None
    with pytest.raises(LocalNodeMustContainPrimaryData):
        node_registry_client.register_node(node_params=node, db_params=db)


# compare unhashable lists, this is slow and hacky but it is only for testing
def _compare_unordered_lists(list1, list2):
    list1_copy = list(list1)
    try:
        for elem in list2:
            list1_copy.remove(elem)
    except ValueError:
        return False
    return not list1_copy
