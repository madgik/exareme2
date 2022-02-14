import subprocess
import uuid

import fasteners
import pytest

from mipengine.node_info_DTOs import NodeInfo
from tests.dev_env_tests.nodes_communication import get_celery_app
from tests.dev_env_tests.nodes_communication import get_celery_task_signature


@pytest.fixture(autouse=True)
def run_tests_sequentially():
    with fasteners.InterProcessLock("semaphore.lock"):
        yield


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
    "node_id, expected_node_info",
    test_cases_get_node_info,
)
def test_get_node_info(node_id, expected_node_info):
    request_id = "test_node_info_" + uuid.uuid4().hex + "_request"
    node_app = get_celery_app(node_id)
    node_info_signature = get_celery_task_signature(node_app, "get_node_info")
    task_response = node_info_signature.delay(request_id=request_id).get()
    node_info = NodeInfo.parse_raw(task_response)

    # Compare id and role. IPs and ports are machine dependent and
    # datasets_per_schema is tested elsewhere.
    assert node_info.id == expected_node_info.id
    assert node_info.role == expected_node_info.role


def setup_data_table_in_db(node_id, datasets_per_schema):
    data_model_id = 0
    dataset_id = 0
    for data_model in datasets_per_schema.keys():
        data_model_id += 1
        dataset_id += 1

        sql_query = f"""INSERT INTO "mipdb_metadata"."data_models" VALUES ({data_model_id}, '{data_model}', '0.1', '{data_model}', 'ENABLED', null);"""
        cmd = f'docker exec -i monetdb-{node_id} mclient db -s "{sql_query}"'
        subprocess.call(cmd, shell=True)
        for dataset_name in datasets_per_schema[data_model]:
            dataset_id += 1
            sql_query = f"""INSERT INTO "mipdb_metadata"."datasets" VALUES ({dataset_id}, {data_model_id}, '{dataset_name}', '{dataset_name}', 'ENABLED', null);"""
            cmd = f'docker exec -i monetdb-{node_id} mclient db -s "{sql_query}"'
            subprocess.call(cmd, shell=True)


# The cleanup task cannot be used because it requires specific table name convention
# that doesn't fit with the initial data table names
def teardown_data_tables_in_db(node_id):
    sql_query = f'DELETE FROM "mipdb_metadata"."datasets";'
    cmd = f"docker exec -i monetdb-{node_id} mclient db -s '{sql_query}'"
    subprocess.call(cmd, shell=True)
    sql_query = f'DELETE FROM "mipdb_metadata"."data_models";'
    cmd = f"docker exec -i monetdb-{node_id} mclient db -s '{sql_query}'"
    subprocess.call(cmd, shell=True)


test_cases_get_node_info_datasets = [
    (
        "globalnode",
        {
            "schema1": [
                "dataset1",
                "dataset2",
                "dataset5",
                "dataset7",
                "dataset15",
            ],
            "schema2": ["dataset3"],
            "schema3": ["dataset4"],
        },
    ),
    (
        "globalnode",
        {
            "schema1": [
                "dataset123",
            ],
            "schema2": [
                "dataset123",
            ],
        },
    ),
    (
        "globalnode",
        {
            "schema1": [],
        },
    ),
    (
        "globalnode",
        {},
    ),
]


@pytest.mark.parametrize(
    "node_id, expected_datasets_per_schema",
    test_cases_get_node_info_datasets,
)
def test_get_node_info_datasets(node_id, expected_datasets_per_schema):
    request_id = "test_node_info_" + uuid.uuid4().hex + "_request"
    setup_data_table_in_db(node_id, expected_datasets_per_schema)
    node_app = get_celery_app(node_id)
    get_node_info_signature = get_celery_task_signature(node_app, "get_node_info")
    task_response = get_node_info_signature.delay(request_id=request_id).get()
    node_info = NodeInfo.parse_raw(task_response)

    assert set(node_info.datasets_per_schema.keys()) == set(
        expected_datasets_per_schema.keys()
    )
    for schema in expected_datasets_per_schema.keys():
        assert set(node_info.datasets_per_schema[schema]) == set(
            expected_datasets_per_schema[schema]
        )

    teardown_data_tables_in_db(node_id)
