import uuid

import pytest

from exareme2.datatypes import DType
from exareme2.exceptions import IncompatibleSchemasMergeException
from exareme2.exceptions import TablesNotFound
from exareme2.node_tasks_DTOs import ColumnInfo
from exareme2.node_tasks_DTOs import TableInfo
from exareme2.node_tasks_DTOs import TableSchema
from exareme2.node_tasks_DTOs import TableType
from tests.standalone_tests.conftest import MONETDB_LOCALNODE1_PORT
from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.conftest import insert_data_to_db
from tests.standalone_tests.nodes_communication_helper import get_celery_task_signature
from tests.standalone_tests.std_output_logger import StdOutputLogger

create_table_task_signature = get_celery_task_signature("create_table")
create_merge_table_task_signature = get_celery_task_signature("create_merge_table")
get_merge_tables_task_signature = get_celery_task_signature("get_merge_tables")


@pytest.fixture(autouse=True)
def request_id():
    return "testmergetables" + uuid.uuid4().hex + "request"


@pytest.fixture(autouse=True)
def context_id(request_id):
    context_id = "testmergetables" + uuid.uuid4().hex

    yield context_id


def create_two_column_table(request_id, context_id, table_id: int, celery_app):
    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
        ]
    )
    async_result = celery_app.queue_task(
        task_signature=create_table_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=f"{context_id}table{table_id}",
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    )
    table_info = TableInfo.parse_raw(
        celery_app.get_result(
            async_result=async_result,
            logger=StdOutputLogger(),
            timeout=TASKS_TIMEOUT,
        )
    )
    return table_info


def create_three_column_table_with_data(
    request_id, context_id, table_id: int, celery_app, db_cursor
):
    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )
    async_result = celery_app.queue_task(
        task_signature=create_table_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=f"{context_id}table{table_id}",
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    )
    table_info = TableInfo.parse_raw(
        celery_app.get_result(
            async_result=async_result,
            logger=StdOutputLogger(),
            timeout=TASKS_TIMEOUT,
        )
    )

    values = [[1, 0.1, "test1"], [2, 0.2, "test2"], [3, 0.3, "test3"]]
    insert_data_to_db(table_info.name, values, db_cursor)

    return table_info


@pytest.mark.slow
def test_create_and_get_merge_table(
    request_id,
    context_id,
    localnode1_node_service,
    localnode1_celery_app,
    localnode1_db_cursor,
):
    tables_to_be_merged = [
        create_three_column_table_with_data(
            request_id, context_id, count, localnode1_celery_app, localnode1_db_cursor
        )
        for count in range(0, 5)
    ]
    async_result = localnode1_celery_app.queue_task(
        task_signature=create_merge_table_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        table_infos_json=[table_info.json() for table_info in tables_to_be_merged],
    )
    merge_table_info = TableInfo.parse_raw(
        localnode1_celery_app.get_result(
            async_result=async_result,
            logger=StdOutputLogger(),
            timeout=TASKS_TIMEOUT,
        )
    )

    async_result = localnode1_celery_app.queue_task(
        task_signature=get_merge_tables_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
    )
    merge_tables = localnode1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )
    assert merge_table_info.name in merge_tables


@pytest.mark.slow
def test_incompatible_schemas_merge(
    request_id,
    context_id,
    localnode1_node_service,
    localnode1_celery_app,
    localnode1_db_cursor,
):
    incompatible_partition_tables = [
        create_three_column_table_with_data(
            request_id, context_id, 1, localnode1_celery_app, localnode1_db_cursor
        ),
        create_two_column_table(request_id, context_id, 2, localnode1_celery_app),
        create_two_column_table(request_id, context_id, 3, localnode1_celery_app),
        create_three_column_table_with_data(
            request_id, context_id, 4, localnode1_celery_app, localnode1_db_cursor
        ),
    ]
    async_result = localnode1_celery_app.queue_task(
        task_signature=create_merge_table_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        table_infos_json=[
            table_info.json() for table_info in incompatible_partition_tables
        ],
    )

    with pytest.raises(IncompatibleSchemasMergeException):
        localnode1_celery_app.get_result(
            async_result=async_result,
            logger=StdOutputLogger(),
            timeout=TASKS_TIMEOUT,
        )


@pytest.mark.slow
def test_table_cannot_be_found(
    request_id,
    context_id,
    localnode1_node_service,
    localnode1_celery_app,
    localnode1_db_cursor,
):
    not_found_tables = [
        create_three_column_table_with_data(
            request_id, context_id, 1, localnode1_celery_app, localnode1_db_cursor
        ),
        create_three_column_table_with_data(
            request_id, context_id, 2, localnode1_celery_app, localnode1_db_cursor
        ),
        TableInfo(
            name="non_existing_table",
            schema_=TableSchema(
                columns=[ColumnInfo(name="non_existing", dtype=DType.INT)]
            ),
            type_=TableType.NORMAL,
        ),
    ]

    async_result = localnode1_celery_app.queue_task(
        task_signature=create_merge_table_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        table_infos_json=[table_info.json() for table_info in not_found_tables],
    )

    with pytest.raises(TablesNotFound):
        localnode1_celery_app.get_result(
            async_result=async_result,
            logger=StdOutputLogger(),
            timeout=TASKS_TIMEOUT,
        )
