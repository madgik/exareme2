import uuid

import pytest

from exareme2.datatypes import DType
from exareme2.node_communication import ColumnInfo
from exareme2.node_communication import IncompatibleSchemasMergeException
from exareme2.node_communication import TableInfo
from exareme2.node_communication import TableSchema
from exareme2.node_communication import TablesNotFound
from exareme2.node_communication import TableType
from tests.standalone_tests.conftest import COMMON_IP
from tests.standalone_tests.conftest import MONETDB_LOCALNODE1_PORT
from tests.standalone_tests.conftest import MONETDB_LOCALNODE2_PORT
from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.conftest import create_table_in_db
from tests.standalone_tests.conftest import get_table_data_from_db
from tests.standalone_tests.conftest import insert_data_to_db
from tests.standalone_tests.nodes_communication_helper import get_celery_task_signature
from tests.standalone_tests.std_output_logger import StdOutputLogger

create_remote_task_signature = get_celery_task_signature("create_remote_table")
create_merge_table_task_signature = get_celery_task_signature("create_merge_table")
get_merge_tables_task_signature = get_celery_task_signature("get_merge_tables")


@pytest.fixture(autouse=True)
def request_id():
    return "testmergetables" + uuid.uuid4().hex + "request"


@pytest.fixture(autouse=True)
def context_id(request_id):
    context_id = "testmergetables" + uuid.uuid4().hex

    yield context_id


def create_two_column_table(db_cursor, table_name):
    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
        ]
    )
    create_table_in_db(db_cursor, table_name, table_schema)

    return TableInfo(
        name=table_name,
        schema_=table_schema,
        type_=TableType.NORMAL,
    )


def create_three_column_table_with_data(db_cursor, table_name):
    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )
    create_table_in_db(db_cursor, table_name, table_schema)

    values = [[1, 0.1, "test1"], [2, 0.2, "test2"], [3, 0.3, "test3"]]
    insert_data_to_db(table_name, values, db_cursor)

    return TableInfo(
        name=table_name,
        schema_=table_schema,
        type_=TableType.NORMAL,
    )


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
            localnode1_db_cursor, f"normal_{context_id}_{count}"
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
def test_merge_table_incompatible_schemas(
    request_id,
    context_id,
    localnode1_node_service,
    localnode1_celery_app,
    localnode1_db_cursor,
):
    incompatible_partition_tables = [
        create_three_column_table_with_data(
            localnode1_db_cursor, f"normal_{context_id}_1"
        ),
        create_two_column_table(localnode1_db_cursor, f"normal_{context_id}_2"),
        create_two_column_table(localnode1_db_cursor, f"normal_{context_id}_3"),
        create_three_column_table_with_data(
            localnode1_db_cursor, f"normal_{context_id}_4"
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
def test_merge_table_cannot_find_table(
    request_id,
    context_id,
    localnode1_node_service,
    localnode1_celery_app,
    localnode1_db_cursor,
):
    not_found_tables = [
        create_three_column_table_with_data(
            localnode1_db_cursor, f"normal_{context_id}_1"
        ),
        create_three_column_table_with_data(
            localnode1_db_cursor, f"normal_{context_id}_2"
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


@pytest.mark.slow
def test_create_merge_table_on_top_of_remote_tables(
    request_id,
    context_id,
    localnode1_node_service,
    localnode1_celery_app,
    localnode1_db_cursor,
    use_localnode1_database,
    localnode2_node_service,
    localnode2_celery_app,
    localnode2_db_cursor,
    use_localnode2_database,
    globalnode_node_service,
    globalnode_celery_app,
    globalnode_db_cursor,
    use_globalnode_database,
):
    """
    The following method tests that the monetdb concept of remote tables combined by a merge table works properly.
    We are using the create_remote_table and create_merge_table celery in the NODE to create the flow.
    The initial tables are created through a db cursor.
    """
    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )
    initial_table_values = [[1, 0.1, "test1"], [2, 0.2, "test2"], [3, 0.3, "test3"]]
    localnode1_tableinfo = TableInfo(
        name=f"normal_testlocalnode1_{context_id}",
        schema_=table_schema,
        type_=TableType.NORMAL,
    )
    localnode2_tableinfo = TableInfo(
        name=f"normal_testlocalnode2_{context_id}",
        schema_=table_schema,
        type_=TableType.NORMAL,
    )
    create_table_in_db(
        localnode1_db_cursor,
        localnode1_tableinfo.name,
        localnode1_tableinfo.schema_,
        True,
    )
    create_table_in_db(
        localnode2_db_cursor,
        localnode2_tableinfo.name,
        localnode2_tableinfo.schema_,
        True,
    )
    insert_data_to_db(
        localnode1_tableinfo.name, initial_table_values, localnode1_db_cursor
    )
    insert_data_to_db(
        localnode2_tableinfo.name, initial_table_values, localnode2_db_cursor
    )

    # Create remote tables
    local_node_1_monetdb_sock_address = f"{str(COMMON_IP)}:{MONETDB_LOCALNODE1_PORT}"
    local_node_2_monetdb_sock_address = f"{str(COMMON_IP)}:{MONETDB_LOCALNODE2_PORT}"

    async_result = globalnode_celery_app.queue_task(
        task_signature=create_remote_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=localnode1_tableinfo.name,
        table_schema_json=table_schema.json(),
        monetdb_socket_address=local_node_1_monetdb_sock_address,
    )

    globalnode_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    async_result = globalnode_celery_app.queue_task(
        task_signature=create_remote_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=localnode2_tableinfo.name,
        table_schema_json=table_schema.json(),
        monetdb_socket_address=local_node_2_monetdb_sock_address,
    )

    globalnode_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    # Create merge table
    async_result = globalnode_celery_app.queue_task(
        task_signature=create_merge_table_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        table_infos_json=[
            localnode1_tableinfo.json(),
            localnode2_tableinfo.json(),
        ],
    )

    merge_table_info = TableInfo.parse_raw(
        globalnode_celery_app.get_result(
            async_result=async_result,
            logger=StdOutputLogger(),
            timeout=TASKS_TIMEOUT,
        )
    )

    # Validate merge tables contains both remote tables' values
    merge_table_values = get_table_data_from_db(
        globalnode_db_cursor, merge_table_info.name
    )

    column_count = len(initial_table_values[0])
    assert column_count == len(merge_table_values[0])

    row_count = len(initial_table_values)
    assert row_count * 2 == len(
        merge_table_values
    )  # The rows are doubled since we have 2 localnodes with N rows each.
