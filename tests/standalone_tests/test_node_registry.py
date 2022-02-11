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
            "data_model1": [
                "dataset1",
                "dataset2",
                "dataset3",
                "dataset4",
                "dataset5",
            ],
            "data_model2": ["dataset6"],
        },
        "localnode2": {
            "data_model2": [
                "dataset7",
                "dataset8",
                "dataset9",
            ],
        },
        "localnode3": {
            "data_model2": [
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
            db_port=50000,
            datasets_per_data_model=get_nodes_datasets_per_data_model()["localnode1"],
        ),
        NodeInfo(
            id="localnode2",
            role=NodeRole.LOCALNODE,
            ip=mocked_node_addresses[2].split(":")[0],
            port=mocked_node_addresses[2].split(":")[1],
            db_ip="127.0.0.1",
            db_port=50000,
            datasets_per_data_model=get_nodes_datasets_per_data_model()["localnode2"],
        ),
        NodeInfo(
            id="localnode3",
            role=NodeRole.LOCALNODE,
            ip=mocked_node_addresses[2].split(":")[0],
            port=mocked_node_addresses[2].split(":")[1],
            db_ip="127.0.0.1",
            db_port=50000,
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


test_cases_get_nodes_with_any_of_datasets = [
    ("data_model1", ["dataset1", "dataset2"], ["localnode1"]),
    ("data_model1", ["non_existing"], []),
    ("non_existing", ["dataset1"], []),
    ("data_model2", ["dataset6"], ["localnode1"]),
    ("data_model2", ["dataset7"], ["localnode2"]),
    (
        "data_model2",
        ["dataset6", "dataset7", "dataset8", "dataset10"],
        ["localnode1", "localnode2", "localnode3"],
    ),
]


@pytest.mark.parametrize(
    "data_model_code, datasets, node_ids",
    test_cases_get_nodes_with_any_of_datasets,
)
def test_get_nodes_with_any_of_datasets(
    data_model_code, datasets, node_ids, mocked_node_registry
):
    test_nodes = mocked_node_registry.get_nodes_with_any_of_datasets(
        data_model_code, datasets
    )
    assert len(test_nodes) == len(node_ids)
    test_node_ids = [test_node.id for test_node in test_nodes]
    assert set(test_node_ids) == set(node_ids)


test_cases_data_model_exists = [
    ("data_model1", True),
    ("data_model2", True),
    ("non_existing", False),
]


@pytest.mark.parametrize(
    "data_model_code, exists",
    test_cases_data_model_exists,
)
def test_data_model_exists(data_model_code, exists, mocked_node_registry):
    assert mocked_node_registry.data_model_exists(data_model_code) == exists


test_cases_dataset_exists = [
    ("data_model1", "dataset1", True),
    ("data_model1", "dataset5", True),
    ("data_model2", "dataset6", True),
    ("data_model2", "dataset10", True),
    ("non_existing", "dataset6", False),
    ("data_model1", "non_existing", False),
    ("data_model1", "dataset6", False),
]


@pytest.mark.parametrize(
    "data_model_code, dataset, exists",
    test_cases_dataset_exists,
)
def test_dataset_exists(data_model_code, dataset, exists, mocked_node_registry):
    assert mocked_node_registry.dataset_exists(data_model_code, dataset) == exists


test_cases_get_nodes_with_any_of_datasets = [
    ("data_model1", ["dataset1"], ["localnode1"]),
    ("data_model1", ["dataset1", "dataset2"], ["localnode1"]),
    ("data_model1", ["dataset1", "dataset6"], ["localnode1"]),
    ("data_model1", ["dataset1", "dataset7"], ["localnode1"]),
    (
        "data_model2",
        ["dataset1", "dataset7", "dataset10"],
        ["localnode2", "localnode3"],
    ),
    (
        "data_model2",
        ["dataset6", "dataset7", "dataset10"],
        ["localnode1", "localnode2", "localnode3"],
    ),
]


@pytest.mark.parametrize(
    "data_model_code, datasets, expected_node_names",
    test_cases_get_nodes_with_any_of_datasets,
)
def test_get_nodes_with_any_of_datasets(
    data_model_code, datasets, expected_node_names, mocked_node_registry
):
    nodes_info = mocked_node_registry.get_nodes_with_any_of_datasets(
        data_model_code, datasets
    )
    node_names = [node_info.id for node_info in nodes_info]
    node_names.sort()
    expected_node_names.sort()
    assert node_names == expected_node_names


def test_get_all_available_data_models(mocked_node_registry):
    expected_available_data_models = ["data_model1", "data_model2"]
    expected_available_data_models.sort()

    available_data_models = mocked_node_registry.get_all_available_data_models()
    available_data_models.sort()

    assert available_data_models == expected_available_data_models


def test_get_all_available_datasets_per_data_model(mocked_node_registry):
    expected_datasets_per_data_model_code = {
        "data_model1": ["dataset1", "dataset2", "dataset3", "dataset4", "dataset5"],
        "data_model2": ["dataset6", "dataset7", "dataset8", "dataset9", "dataset10"],
    }

    datasets_per_data_model_code = (
        mocked_node_registry.get_all_available_datasets_per_data_model_code()
    )

    assert datasets_per_data_model_code == expected_datasets_per_data_model_code
