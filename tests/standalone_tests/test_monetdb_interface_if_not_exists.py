from unittest.mock import patch

import pytest

from mipengine import AttrDict
from mipengine import DType
from mipengine.node.monetdb_interface.merge_tables import create_merge_table
from mipengine.node.monetdb_interface.monet_db_facade import _execute
from mipengine.node.monetdb_interface.remote_tables import create_remote_table
from mipengine.node.monetdb_interface.tables import create_table
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
                "username": "monetdb",
                "password": "monetdb",
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
    )

    try:
        create_remote_table(
            name=table_name,
            monetdb_socket_address="127.0.0.1:50000",
            schema=table_schema,
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


def test_create_table_if_not_exists_query():
    with patch(
        "mipengine.node.monetdb_interface.monet_db_facade._execute_and_commit"
    ) as execute_and_commit_mock:
        _execute("CREATE TABLE")
        assert (
            execute_and_commit_mock.call_args.args[1].query
            == "CREATE TABLE IF NOT EXISTS"
        )


def test_create_remote_table_if_not_exists_query():
    with patch(
        "mipengine.node.monetdb_interface.monet_db_facade._execute_and_commit"
    ) as execute_and_commit_mock:
        _execute("CREATE REMOTE TABLE")
        assert (
            execute_and_commit_mock.call_args.args[1].query
            == "CREATE REMOTE TABLE IF NOT EXISTS"
        )


def test_create_merge_table_if_not_exists_query():
    with patch(
        "mipengine.node.monetdb_interface.monet_db_facade._execute_and_commit"
    ) as execute_and_commit_mock:
        _execute("CREATE MERGE TABLE sample_table")
        assert (
            execute_and_commit_mock.call_args.args[1].query
            == "DROP TABLE IF EXISTS sample_table; CREATE MERGE TABLE sample_table"
        )


def test_create_view_if_not_exists_query():
    with patch(
        "mipengine.node.monetdb_interface.monet_db_facade._execute_and_commit"
    ) as execute_and_commit_mock:
        _execute("CREATE VIEW")
        assert (
            execute_and_commit_mock.call_args.args[1].query == "CREATE OR REPLACE VIEW"
        )


def test_drop_view_if_exists_query():
    with patch(
        "mipengine.node.monetdb_interface.monet_db_facade._execute_and_commit"
    ) as execute_and_commit_mock:
        _execute("DROP VIEW")
        assert execute_and_commit_mock.call_args.args[1].query == "DROP VIEW IF EXISTS"


def test_drop_table_if_exists_query():
    with patch(
        "mipengine.node.monetdb_interface.monet_db_facade._execute_and_commit"
    ) as execute_and_commit_mock:
        _execute("DROP TABLE")
        assert execute_and_commit_mock.call_args.args[1].query == "DROP TABLE IF EXISTS"


def test_drop_function_if_exists_query():
    with patch(
        "mipengine.node.monetdb_interface.monet_db_facade._execute_and_commit"
    ) as execute_and_commit_mock:
        _execute("DROP FUNCTION")
        assert (
            execute_and_commit_mock.call_args.args[1].query == "DROP FUNCTION IF EXISTS"
        )
