import uuid

import pytest

from mipengine.datatypes import DType
from mipengine.node_exceptions import IncompatibleSchemasMergeException
from mipengine.node_exceptions import TablesNotFound
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import TableSchema
from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.nodes_communication_helper import get_celery_task_signature
from tests.standalone_tests.std_output_logger import StdOutputLogger

create_table_task_signature = get_celery_task_signature("create_table")
create_merge_table_task_signature = get_celery_task_signature("create_merge_table")
insert_data_to_table_task_signature = get_celery_task_signature("insert_data_to_table")
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
    table_name = celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )
    return table_name


def create_three_column_table_with_data(
    request_id, context_id, table_id: int, celery_app
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
    table_name = celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    values = [[1, 0.1, "test1"], [2, 0.2, "test2"], [3, 0.3, "test3"]]
    async_result = celery_app.queue_task(
        task_signature=insert_data_to_table_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=table_name,
        values=values,
    )
    celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    return table_name


def test_create_and_get_merge_table(
    request_id,
    context_id,
    localnode1_node_service,
    localnode1_celery_app,
):
    tables_to_be_merged = [
        create_three_column_table_with_data(
            request_id, context_id, count, localnode1_celery_app
        )
        for count in range(0, 5)
    ]
    async_result = localnode1_celery_app.queue_task(
        task_signature=create_merge_table_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        table_names=tables_to_be_merged,
    )
    merge_table_1_name = localnode1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
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
    assert merge_table_1_name in merge_tables


@pytest.mark.skip(reason="https://team-1617704806227.atlassian.net/browse/MIP-649")
def test_incompatible_schemas_merge(
    request_id,
    context_id,
    localnode1_node_service,
    localnode1_celery_app,
):
    with pytest.raises(IncompatibleSchemasMergeException):
        incompatible_partition_tables = [
            create_three_column_table_with_data(
                request_id, context_id, 1, localnode1_celery_app
            ),
            create_two_column_table(request_id, context_id, 2, localnode1_celery_app),
            create_two_column_table(request_id, context_id, 3, localnode1_celery_app),
            create_three_column_table_with_data(
                request_id, context_id, 4, localnode1_celery_app
            ),
        ]
        async_result = localnode1_celery_app.queue_task(
            task_signature=create_merge_table_task_signature,
            logger=StdOutputLogger(),
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            table_names=incompatible_partition_tables,
        )
        localnode1_celery_app.get_result(
            async_result=async_result,
            logger=StdOutputLogger(),
            timeout=TASKS_TIMEOUT,
        )


def test_table_cannot_be_found(
    request_id,
    context_id,
    localnode1_node_service,
    localnode1_celery_app,
):
    with pytest.raises(TablesNotFound):
        not_found_tables = [
            create_three_column_table_with_data(
                request_id, context_id, 1, localnode1_celery_app
            ),
            create_three_column_table_with_data(
                request_id, context_id, 2, localnode1_celery_app
            ),
            "non_existing_table",
        ]

        async_result = localnode1_celery_app.queue_task(
            task_signature=create_merge_table_task_signature,
            logger=StdOutputLogger(),
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            table_names=not_found_tables,
        )
        localnode1_celery_app.get_result(
            async_result=async_result,
            logger=StdOutputLogger(),
            timeout=TASKS_TIMEOUT,
        )
