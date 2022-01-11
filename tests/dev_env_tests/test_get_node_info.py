import pytest
from mipengine.node_info_DTOs import NodeInfo
from tests.dev_env_tests.nodes_communication import get_celery_app
from tests.dev_env_tests.nodes_communication import get_celery_task_signature

test_cases_get_node_info = [
    (
        "globalnode",
        NodeInfo(
            id="globalnode",
            role="GLOBALNODE",
            ip="172.17.0.1",
            port=5670,
            db_ip="172.17.0.1",
            db_port=50000,
            datasets_per_schema={},
        ),
    ),
    (
        "localnode1",
        NodeInfo(
            id="localnode1",
            role="LOCALNODE",
            ip="172.17.0.1",
            port=5671,
            db_ip="172.17.0.1",
            db_port=50001,
            datasets_per_schema={},
        ),
    ),
]


@pytest.mark.parametrize(
    "node_id, proper_node_info",
    test_cases_get_node_info,
)
def get_node_info(node_id, proper_node_info):
    node_app = get_celery_app(node_id)
    node_info_signature = get_celery_task_signature(node_app, "get_node_info")
    task_response = node_info_signature.delay().get()
    node_info = NodeInfo.parse_raw(task_response)

    # Compare all the NodeInfo but the datasets_per_schema, it's tested separately
    node_info.datasets_per_schema = None
    proper_node_info.datasets_per_schema = None
    assert node_info == proper_node_info


test_cases_get_node_info_datasets = [
    (
        "localnode1",
        {
            "tbi": ["dummy_tbi"],
            "dementia": ["ppmi", "edsd", "desd-synthdata"],
        },
    ),
    (
        "globalnode",
        {},
    ),
]


@pytest.mark.parametrize(
    "node_id, proper_datasets_per_schema",
    test_cases_get_node_info_datasets,
)
def test_get_node_info_datasets(node_id, proper_datasets_per_schema):
    node_app = get_celery_app(node_id)
    get_node_info_signature = get_celery_task_signature(node_app, "get_node_info")
    task_response = get_node_info_signature.delay().get()
    node_info = NodeInfo.parse_raw(task_response)

    assert set(node_info.datasets_per_schema.keys()) == set(
        proper_datasets_per_schema.keys()
    )
    for schema in proper_datasets_per_schema.keys():
        assert set(node_info.datasets_per_schema[schema]) == set(
            proper_datasets_per_schema[schema]
        )
