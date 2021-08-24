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


def get_mocked_node_info() -> List[NodeInfo]:
    return [
        NodeInfo(
            id="globalnode",
            role=NodeRole.GLOBALNODE,
            ip=mocked_node_addresses[0].split(":")[0],
            port=mocked_node_addresses[0].split(":")[1],
            db_ip="127.0.0.1",
            db_port=50000,
            datasets_per_schema=None,
        ),
        NodeInfo(
            id="localnode1",
            role=NodeRole.LOCALNODE,
            ip=mocked_node_addresses[1].split(":")[0],
            port=mocked_node_addresses[1].split(":")[1],
            db_ip="127.0.0.1",
            db_port=50000,
            datasets_per_schema={
                "schema1": [
                    "dataset1",
                    "dataset2",
                    "dataset3",
                    "dataset4",
                    "dataset5",
                ],
                "schema2": ["dataset6"],
            },
        ),
        NodeInfo(
            id="localnode2",
            role=NodeRole.LOCALNODE,
            ip=mocked_node_addresses[2].split(":")[0],
            port=mocked_node_addresses[2].split(":")[1],
            db_ip="127.0.0.1",
            db_port=50000,
            datasets_per_schema={
                "schema2": [
                    "dataset7",
                    "dataset8",
                    "dataset9",
                ],
            },
        ),
        NodeInfo(
            id="localnode3",
            role=NodeRole.LOCALNODE,
            ip=mocked_node_addresses[2].split(":")[0],
            port=mocked_node_addresses[2].split(":")[1],
            db_ip="127.0.0.1",
            db_port=50000,
            datasets_per_schema={
                "schema2": [
                    "dataset10",
                ],
            },
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
    ("schema1", ["dataset1", "dataset2"], ["localnode1"]),
    ("schema1", ["non_existing"], []),
    ("non_existing", ["dataset1"], []),
    ("schema2", ["dataset6"], ["localnode1"]),
    ("schema2", ["dataset7"], ["localnode2"]),
    (
        "schema2",
        ["dataset6", "dataset7", "dataset8", "dataset10"],
        ["localnode1", "localnode2", "localnode3"],
    ),
]


@pytest.mark.parametrize(
    "schema, datasets, node_ids",
    test_cases_get_nodes_with_any_of_datasets,
)
def test_get_nodes_with_any_of_datasets(
    schema, datasets, node_ids, mocked_node_registry
):
    test_nodes = mocked_node_registry.get_nodes_with_any_of_datasets(schema, datasets)
    assert len(test_nodes) == len(node_ids)
    test_node_ids = [test_node.id for test_node in test_nodes]
    assert set(test_node_ids) == set(node_ids)


test_cases_schema_exists = [
    ("schema1", True),
    ("schema2", True),
    ("non_existing", False),
]


@pytest.mark.parametrize(
    "schema, exists",
    test_cases_schema_exists,
)
def test_schema_exists(schema, exists, mocked_node_registry):
    assert mocked_node_registry.schema_exists(schema) == exists


test_cases_dataset_exists = [
    ("schema1", "dataset1", True),
    ("schema1", "dataset5", True),
    ("schema2", "dataset6", True),
    ("schema2", "dataset10", True),
    ("non_existing", "dataset6", False),
    ("schema1", "non_existing", False),
    ("schema1", "dataset6", False),
]


@pytest.mark.parametrize(
    "schema, dataset, exists",
    test_cases_dataset_exists,
)
def test_dataset_exists(schema, dataset, exists, mocked_node_registry):
    assert mocked_node_registry.dataset_exists(schema, dataset) == exists
