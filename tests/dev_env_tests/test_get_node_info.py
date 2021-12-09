import fasteners
import pytest
import subprocess

from mipengine.node.monetdb_interface.common_actions import (
    convert_schema_to_sql_query_format,
)
from mipengine.node_info_DTOs import NodeInfo
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.datatypes import DType
from mipengine.node_tasks_DTOs import TableSchema
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
    "node_id, proper_node_info",
    test_cases_get_node_info,
)
def test_get_node_info(node_id, proper_node_info):
    node_app = get_celery_app(node_id)
    node_info_signature = get_celery_task_signature(node_app, "get_node_info")
    task_response = node_info_signature.delay().get()
    node_info = NodeInfo.parse_raw(task_response)

    # Compare all the NodeInfo but the datasets_per_schema, it's tested separately
    node_info.datasets_per_schema = None
    proper_node_info.datasets_per_schema = None
    assert node_info == proper_node_info


def setup_data_table_in_db(node_id, datasets_per_schema):
    tables_schema = TableSchema(
        columns=[
            ColumnInfo(name="dataset", dtype=DType.STR),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )
    for schema_name in datasets_per_schema.keys():
        table_name = f"{schema_name}_data"
        create_data_table_in_db(node_id, table_name, tables_schema)
        table_values = []
        for dataset_name in datasets_per_schema[schema_name]:
            table_values.append([dataset_name, 0.1, "demodata"])
        if table_values:
            add_values_to_table(node_id, table_name, table_values)


# The create_table task cannot be used because it requires specific table name convention
# that doesn't fit with the initial data table names
def create_data_table_in_db(node_id, table_name, table_schema):
    sql_columns_schema_query = convert_schema_to_sql_query_format(table_schema)
    sql_query = (
        f"DROP TABLE IF EXISTS {table_name}; "
        f"CREATE TABLE {table_name} ({sql_columns_schema_query});"
    )
    cmd = f"docker exec -i monetdb-{node_id} mclient db -s '{sql_query}'"
    subprocess.call(cmd, shell=True)


def add_values_to_table(node_id, table_name, table_values):
    node_app = get_celery_app(node_id)
    node_insert_data_to_table = get_celery_task_signature(
        node_app, "insert_data_to_table"
    )

    node_insert_data_to_table.delay(table_name=table_name, values=table_values).get()


# The cleanup task cannot be used because it requires specific table name convention
# that doesn't fit with the initial data table names
def teardown_data_tables_in_db(node_id, datasets_per_schema):
    for schema_name in datasets_per_schema.keys():
        sql_query = f"DROP TABLE {schema_name}_data; "
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
    "node_id, proper_datasets_per_schema",
    test_cases_get_node_info_datasets,
)
def test_get_node_info_datasets(node_id, proper_datasets_per_schema):
    setup_data_table_in_db(node_id, proper_datasets_per_schema)
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

    teardown_data_tables_in_db(node_id, proper_datasets_per_schema)
