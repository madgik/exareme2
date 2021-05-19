import json
import uuid

import pytest

from mipengine.common.node_tasks_DTOs import ColumnInfo, TableSchema
from tests.integration_tests.nodes_communication import get_celery_app
from tests.integration_tests.nodes_communication import get_celery_task_signature

local_node = get_celery_app("localnode1")
local_node_create_table = get_celery_task_signature(local_node, "create_table")
local_node_get_tables = get_celery_task_signature(local_node, "get_tables")
local_node_get_table_schema = get_celery_task_signature(local_node, "get_table_schema")
local_node_get_table_data = get_celery_task_signature(local_node, "get_table_data")
local_node_create_merge_table = get_celery_task_signature(
    local_node, "create_merge_table"
)
local_node_get_merge_tables = get_celery_task_signature(local_node, "get_merge_tables")
local_node_create_pathology_view = get_celery_task_signature(
    local_node, "create_pathology_view"
)
local_node_get_views = get_celery_task_signature(local_node, "get_views")
local_node_create_remote_table = get_celery_task_signature(
    local_node, "create_remote_table"
)
local_node_get_remote_tables = get_celery_task_signature(
    local_node, "get_remote_tables"
)
local_node_cleanup = get_celery_task_signature(local_node, "clean_up")

context_id = "HISTOGRAMS"


@pytest.fixture(autouse=True)
def cleanup_tables():
    yield

    local_node_cleanup.delay(context_id=context_id.lower()).get()


def test_sql_injection_get_tables():
    with pytest.raises(ValueError):
        local_node_get_tables.delay(context_id=');"').get()


def test_sql_injection_get_table_schema():
    with pytest.raises(ValueError):
        local_node_get_table_schema.delay(
            table_name="Robert'); DROP TABLE data; --"
        ).get()


def test_sql_injection_create_table_context_id():
    with pytest.raises(ValueError):
        invalid_context_id = "Robert'); DROP TABLE data; --"
        schema = TableSchema(
            [
                ColumnInfo("col1", "int"),
                ColumnInfo("col2", "real"),
                ColumnInfo("col3", "text"),
            ]
        )
        json_schema = schema.to_json()
        local_node_create_table.delay(
            context_id=invalid_context_id,
            command_id=str(uuid.uuid1()).replace("-", ""),
            schema_json=json_schema,
        ).get()


def test_sql_injection_create_table_tableschema_name():
    with pytest.raises(ValueError):
        schema = TableSchema(
            [
                ColumnInfo("Robert'); DROP TABLE data; --", "INT"),
                ColumnInfo("col2", "real"),
                ColumnInfo("col3", "text"),
            ]
        )
        json_schema = schema.to_json()
        local_node_create_table.delay(
            context_id=context_id,
            command_id=str(uuid.uuid1()).replace("-", ""),
            schema_json=json_schema,
        ).get()


def test_sql_injection_get_merge_tables():
    with pytest.raises(ValueError):
        local_node_get_merge_tables.delay(
            context_id="Robert'); DROP TABLE data; --"
        ).get()


def test_sql_injection_create_merge_table_table_names():
    with pytest.raises(ValueError):
        local_node_create_merge_table.delay(
            context_id=context_id,
            command_id=str(uuid.uuid1()).replace("-", ""),
            table_names=["Robert'); DROP TABLE data; --"],
        ).get()


def test_sql_injection_get_views():
    with pytest.raises(ValueError):
        local_node_get_views.delay(context_id="Robert'); DROP TABLE data; --").get()


def test_sql_injection_get_view_schema():
    with pytest.raises(ValueError):
        local_node_get_table_schema.delay(
            table_name="Robert'); DROP TABLE data; --"
        ).get()


def test_sql_injection_create_view_columns():
    with pytest.raises(ValueError):
        columns = [
            "Robert'); DROP TABLE data; --",
            "age_value",
            "gcs_motor_response_scale",
            "pupil_reactivity_right_eye_result",
        ]
        datasets = ["edsd"]
        pathology = "tbi"
        local_node_create_pathology_view.delay(
            context_id=context_id,
            command_id=str(uuid.uuid1()).replace("-", ""),
            pathology=pathology,
            datasets=datasets,
            columns=columns,
            filters_json="filters_json",
        ).get()


def test_sql_injection_create_view_datasets():
    with pytest.raises(ValueError):
        columns = [
            "dataset",
            "age_value",
            "gcs_motor_response_scale",
            "pupil_reactivity_right_eye_result",
        ]
        datasets = ["Robert'); DROP TABLE data; --"]
        pathology = "tbi"
        local_node_create_pathology_view.delay(
            context_id=context_id,
            command_id=str(uuid.uuid1()).replace("-", ""),
            pathology=pathology,
            datasets=datasets,
            columns=columns,
            filters_json="",
        ).get()


def test_sql_injection_get_remote_tables():
    with pytest.raises(ValueError):
        local_node_get_remote_tables.delay(
            context_id="Robert'); DROP TABLE data; --"
        ).get()
