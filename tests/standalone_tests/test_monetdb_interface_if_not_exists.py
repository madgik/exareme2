from unittest.mock import patch

import pytest

from mipengine import AttrDict
from mipengine import DType
from mipengine.node.monetdb_interface.common_actions import get_table_data
from mipengine.node.monetdb_interface.merge_tables import create_merge_table
from mipengine.node.monetdb_interface.monet_db_facade import _create_idempotent_query
from mipengine.node.monetdb_interface.monet_db_facade import _db_execute_udf
from mipengine.node.monetdb_interface.remote_tables import create_remote_table
from mipengine.node.monetdb_interface.tables import create_table
from mipengine.node.monetdb_interface.tables import insert_data_to_table
from mipengine.node.monetdb_interface.views import create_view
from mipengine.node.node_logger import init_logger
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import TableInfo
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import TableType
from tests.standalone_tests.conftest import COMMON_IP
from tests.standalone_tests.conftest import MONETDB_LOCALNODE1_PORT


@pytest.fixture(scope="module", autouse=True)
def patch_node_logger():
    current_task = AttrDict({"request": {"id": "1234"}})

    with patch(
        "mipengine.node.node_logger.node_config",
        AttrDict(
            {
                "log_level": "DEBUG",
                "role": "localnode",
                "identifier": "localnodetmp",
            },
        ),
    ), patch(
        "mipengine.node.node_logger.task_loggers",
        {"1234": init_logger("1234")},
    ), patch(
        "mipengine.node.node_logger.current_task",
        current_task,
    ):
        yield


@pytest.fixture(autouse=True, scope="module")
def patch_node_config():
    node_config = AttrDict(
        {
            "monetdb": {
                "ip": COMMON_IP,
                "port": MONETDB_LOCALNODE1_PORT,
                "database": "db",
                "username": "executor",
                "password": "executor",
            },
            "celery": {
                "tasks_timeout": 60,
                "run_udf_task_timeout": 120,
                "worker_concurrency": 1,
            },
        }
    )

    with patch(
        "mipengine.node.monetdb_interface.monet_db_facade.node_config", node_config
    ):
        yield


@pytest.fixture(scope="function")
def create_sample_table(
    use_localnode1_database,
    localnode1_db_cursor,
) -> TableInfo:
    info = TableInfo(
        name="table_test_tables_if_not_exists",
        schema_=TableSchema(
            columns=[ColumnInfo(name="sample_column", dtype=DType.INT)]
        ),
        type_=TableType.NORMAL,
    )
    localnode1_db_cursor.execute(
        f"CREATE TABLE {info.name} ( {info.column_names[0]} {info.schema_.columns[0].dtype.value})"
    )
    localnode1_db_cursor.execute(
        f"INSERT INTO {info.name} ( {info.column_names[0]} ) VALUES ( 10 )"
    )
    return info


@pytest.mark.slow
def test_create_view_if_not_exists(
    use_localnode1_database,
    create_sample_table,
):
    table_info = create_sample_table
    view_name = "view_test_create_view_if_not_exists"

    create_view(
        view_name=view_name,
        table_name=table_info.name,
        columns=table_info.column_names,
        filters=None,
        minimum_row_count=0,
        check_min_rows=False,
    )

    try:
        create_view(
            view_name=view_name,
            table_name=table_info.name,
            columns=table_info.column_names,
            filters=None,
            minimum_row_count=0,
            check_min_rows=False,
        )
    except Exception as exc:
        pytest.fail(f"A view should be able to be recreated. Exception: \n{str(exc)} ")


@pytest.mark.slow
def test_create_table_if_not_exists(
    use_localnode1_database,
):
    table_name = "test_create_table_if_not_exists"
    table_schema = TableSchema(
        columns=[ColumnInfo(name="sample_column", dtype=DType.INT)]
    )
    create_table(
        table_name=table_name,
        table_schema=table_schema,
    )

    try:
        create_table(
            table_name=table_name,
            table_schema=table_schema,
        )
    except Exception as exc:
        pytest.fail(f"A table should be able to be recreated. Exception: \n{str(exc)} ")


@pytest.mark.slow
def test_insert_into_table_if_empty(
    use_localnode1_database,
):
    table_name = "test_insert_into_table_if_empty"
    table_schema = TableSchema(
        columns=[ColumnInfo(name="sample_column", dtype=DType.INT)]
    )
    create_table(
        table_name=table_name,
        table_schema=table_schema,
    )

    insert_data_to_table(
        table_name=table_name,
        table_values=[[1], [2]],
    )
    insert_data_to_table(
        table_name=table_name,
        table_values=[[3], [4]],
    )
    [column] = get_table_data(table_name)
    data = column.data
    if data != [3, 4]:
        pytest.fail("Each insert into should clear and re-add the values")


@pytest.mark.slow
def test_create_remote_table_if_not_exists(
    use_localnode1_database,
):
    table_name = "test_create_remote_table_if_not_exists"
    table_schema = TableSchema(
        columns=[ColumnInfo(name="sample_column", dtype=DType.INT)]
    )
    create_remote_table(
        name=table_name,
        monetdb_socket_address="127.0.0.1:50000",
        schema=table_schema,
        username="executor",
        password="executor",
    )

    try:
        create_remote_table(
            name=table_name,
            monetdb_socket_address="127.0.0.1:50000",
            schema=table_schema,
            username="executor",
            password="executor",
        )
    except Exception as exc:
        pytest.fail(
            f"A remote table should be able to be recreated. Exception: \n{str(exc)} "
        )


@pytest.mark.slow
def test_create_merge_table_if_not_exists(
    use_localnode1_database,
    create_sample_table,
):
    sample_table_info = create_sample_table
    table_name = "test_create_merge_table_if_not_exists"
    table_schema = TableSchema(
        columns=[ColumnInfo(name="sample_column", dtype=DType.INT)]
    )
    create_merge_table(
        table_name=table_name,
        table_schema=table_schema,
        merge_table_names=[sample_table_info.name],
    )

    try:
        create_merge_table(
            table_name=table_name,
            table_schema=table_schema,
            merge_table_names=[sample_table_info.name],
        )
    except Exception as exc:
        pytest.fail(
            f"A merge table should be able to be recreated. Exception: \n{str(exc)} "
        )


def test_udf_execution_query_should_contain_only_one_query():
    with pytest.raises(ValueError):
        _db_execute_udf(query="INSERT INTO table1 func1();INSERT INTO table2 func2();")


def get_idempotent_query_cases():
    return [
        pytest.param(
            "INSERT INTO table1 values ();INSERT INTO table2 values ();",
            "DELETE FROM table1;INSERT INTO table1 values ();DELETE FROM table2;INSERT INTO table2 values ();",
            id="insert into if values are not present",
        ),
        pytest.param(
            "CREATE TABLE table1 ();CREATE TABLE table2 ();",
            "CREATE TABLE IF NOT EXISTS table1 ();CREATE TABLE IF NOT EXISTS table2 ();",
            id="create table if not exists query",
        ),
        pytest.param(
            "CREATE REMOTE TABLE table1 ();CREATE REMOTE TABLE table2 ();",
            "CREATE REMOTE TABLE IF NOT EXISTS table1 ();CREATE REMOTE TABLE IF NOT EXISTS table2 ();",
            id="create remote table if not exists query",
        ),
        pytest.param(
            "CREATE MERGE TABLE table1 ();CREATE MERGE TABLE table2 ();",
            "DROP TABLE IF EXISTS table1;CREATE MERGE TABLE table1 ();DROP TABLE IF EXISTS table2;CREATE MERGE TABLE table2 ();",
            id="create merge table if not exists query",
        ),
        pytest.param(
            "CREATE VIEW view1 as ();CREATE VIEW view2 as ();",
            "CREATE OR REPLACE VIEW view1 as ();CREATE OR REPLACE VIEW view2 as ();",
            id="mutliple create view if not exists query",
        ),
        pytest.param(
            "DROP VIEW view1;DROP VIEW view2;",
            "DROP VIEW IF EXISTS view1;DROP VIEW IF EXISTS view2;",
            id="drop view if exists query",
        ),
        pytest.param(
            "DROP TABLE table1;DROP TABLE table2;",
            "DROP TABLE IF EXISTS table1;DROP TABLE IF EXISTS table2;",
            id="mutliple drop table if exists query",
        ),
        pytest.param(
            "DROP FUNCTION func1;DROP FUNCTION func2;",
            "DROP FUNCTION IF EXISTS func1;DROP FUNCTION IF EXISTS func2;",
            id="drop function if exists",
        ),
        pytest.param(
            "CREATE TABLE sample_table (col1);"
            "INSERT INTO sample_table VALUES(1);"
            "CREATE REMOTE TABLE sample_remote_table;"
            "CREATE MERGE TABLE sample_merge_table (col1);"
            "CREATE VIEW sample_view as ();"
            "CREATE FUNCTION sample_func stuff;"
            "DROP VIEW sample_view;"
            "DROP TABLE sample_table;"
            "DROP TABLE sample_remote_table;"
            "DROP TABLE sample_merge_table;"
            "DROP FUNCTION sample_func;",
            "CREATE TABLE IF NOT EXISTS sample_table (col1);"
            "DELETE FROM sample_table;"
            "INSERT INTO sample_table VALUES(1);"
            "CREATE REMOTE TABLE IF NOT EXISTS sample_remote_table;"
            "DROP TABLE IF EXISTS sample_merge_table;"
            "CREATE MERGE TABLE sample_merge_table (col1);"
            "CREATE OR REPLACE VIEW sample_view as ();"
            "CREATE OR REPLACE FUNCTION sample_func stuff;"
            "DROP VIEW IF EXISTS sample_view;"
            "DROP TABLE IF EXISTS sample_table;"
            "DROP TABLE IF EXISTS sample_remote_table;"
            "DROP TABLE IF EXISTS sample_merge_table;"
            "DROP FUNCTION IF EXISTS sample_func;",
            id="multiple different queries",
        ),
    ]


@pytest.mark.parametrize(
    "original_query,excpected_idempotent_query",
    get_idempotent_query_cases(),
)
def test_create_idempotent_query(original_query: str, excpected_idempotent_query: str):
    assert _create_idempotent_query(original_query) == excpected_idempotent_query
