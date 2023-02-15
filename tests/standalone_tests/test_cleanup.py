import uuid as uuid

import pytest

from mipengine.datatypes import DType
from mipengine.node.monetdb_interface.common_actions import _get_drop_tables_query
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import TableInfo
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import TableType
from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.nodes_communication_helper import get_celery_task_signature
from tests.standalone_tests.std_output_logger import StdOutputLogger

create_table_task_signature = get_celery_task_signature("create_table")
get_tables_task_signature = get_celery_task_signature("get_tables")
clean_up_task_signature = get_celery_task_signature("cleanup")


@pytest.fixture(autouse=True)
def request_id():
    return "testcleanup" + uuid.uuid4().hex + "request"


@pytest.fixture(autouse=True)
def context_id(request_id):
    context_id = "testcleanup" + uuid.uuid4().hex

    yield context_id


@pytest.mark.slow
def test_create_and_find_tables(
    request_id,
    context_id,
    localnode1_node_service,
    localnode1_celery_app,
):
    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )
    for _ in range(10):
        async_result = localnode1_celery_app.queue_task(
            task_signature=create_table_task_signature,
            logger=StdOutputLogger(),
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            schema_json=table_schema.json(),
        )
        TableInfo.parse_raw(
            localnode1_celery_app.get_result(
                async_result=async_result,
                logger=StdOutputLogger(),
                timeout=TASKS_TIMEOUT,
            )
        )
    async_result = localnode1_celery_app.queue_task(
        task_signature=get_tables_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
    )
    tables = localnode1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    assert len(tables) == 10

    async_result = localnode1_celery_app.queue_task(
        task_signature=clean_up_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
    )
    localnode1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    async_result = localnode1_celery_app.queue_task(
        task_signature=get_tables_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
    )
    tables = localnode1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )
    assert len(tables) == 0


all_cases = [
    (
        {
            TableType.NORMAL: ["table1"],
            TableType.VIEW: ["view_table1"],
            TableType.REMOTE: ["remote_table1"],
            TableType.MERGE: ["merge_table1"],
        },
        "DROP TABLE merge_table1;DROP TABLE remote_table1;DROP VIEW view_table1;DROP TABLE table1;",
    ),
    (
        {
            TableType.NORMAL: [],
            TableType.VIEW: ["view_table1"],
            TableType.REMOTE: ["remote_table1"],
            TableType.MERGE: ["merge_table1"],
        },
        "DROP TABLE merge_table1;DROP TABLE remote_table1;DROP VIEW view_table1;",
    ),
    (
        {
            TableType.NORMAL: ["table1"],
            TableType.VIEW: [],
            TableType.REMOTE: ["remote_table1"],
            TableType.MERGE: ["merge_table1"],
        },
        "DROP TABLE merge_table1;DROP TABLE remote_table1;DROP TABLE table1;",
    ),
    (
        {
            TableType.NORMAL: ["table1"],
            TableType.VIEW: ["view_table1"],
            TableType.REMOTE: [],
            TableType.MERGE: ["merge_table1"],
        },
        "DROP TABLE merge_table1;DROP VIEW view_table1;DROP TABLE table1;",
    ),
    (
        {
            TableType.NORMAL: ["table1"],
            TableType.VIEW: ["view_table1"],
            TableType.MERGE: [],
            TableType.REMOTE: ["remote_table1"],
        },
        "DROP TABLE remote_table1;DROP VIEW view_table1;DROP TABLE table1;",
    ),
    (
        {
            TableType.VIEW: [],
            TableType.NORMAL: [],
            TableType.REMOTE: ["remote_table1"],
            TableType.MERGE: ["merge_table1"],
        },
        "DROP TABLE merge_table1;DROP TABLE remote_table1;",
    ),
    (
        {
            TableType.NORMAL: ["table1"],
            TableType.REMOTE: [],
            TableType.MERGE: [],
            TableType.VIEW: ["view_table1"],
        },
        "DROP VIEW view_table1;DROP TABLE table1;",
    ),
    (
        {
            TableType.NORMAL: [],
            TableType.VIEW: [],
            TableType.REMOTE: [],
            TableType.MERGE: ["merge_table1"],
        },
        "DROP TABLE merge_table1;",
    ),
    (
        {
            TableType.VIEW: [],
            TableType.REMOTE: [],
            TableType.MERGE: [],
            TableType.NORMAL: [],
        },
        "",
    ),
]


@pytest.mark.parametrize("table_names_by_type,expected_query", all_cases)
def test_get_drop_tables_query(table_names_by_type, expected_query):
    assert expected_query == _get_drop_tables_query(table_names_by_type)
