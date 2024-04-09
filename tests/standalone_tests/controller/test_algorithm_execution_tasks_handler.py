import random

import pytest

from exareme2 import DType
from exareme2.worker_communication import ColumnInfo
from exareme2.worker_communication import TableSchema

COMMON_TASKS_REQUEST_ID = "rqst1"


@pytest.fixture
def test_table_params():
    command_id = "cmndid1"
    schema = TableSchema(
        columns=[
            ColumnInfo(name="var1", dtype=DType.INT),
            ColumnInfo(name="var2", dtype=DType.STR),
        ]
    )
    return {"command_id": command_id, "schema": schema}


@pytest.mark.slow
def test_create_table(
    localworker1_tasks_handler, use_localworker1_database, test_table_params
):
    context_id = get_a_random_context_id()
    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]

    table_info = localworker1_tasks_handler.create_table(
        context_id=context_id,
        command_id=command_id,
        schema=schema,
    )

    assert str(table_info.type_) == "NORMAL"
    assert table_info.context_id == context_id
    assert table_info.command_id == command_id


@pytest.mark.slow
def test_get_tables(
    localworker1_tasks_handler, use_localworker1_database, test_table_params
):
    context_id = get_a_random_context_id()
    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]
    table_info = localworker1_tasks_handler.create_table(
        context_id=context_id,
        command_id=command_id,
        schema=schema,
    )
    tables = localworker1_tasks_handler.get_tables(context_id=context_id)

    assert table_info.name in tables


@pytest.mark.slow
def test_get_table_schema(
    localworker1_tasks_handler, use_localworker1_database, test_table_params
):
    context_id = get_a_random_context_id()
    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]
    table_info = localworker1_tasks_handler.create_table(
        context_id=context_id,
        command_id=command_id,
        schema=schema,
    )
    assert table_info.schema_ == schema


def get_a_random_context_id() -> str:
    return str(random.randint(1, 99999))
