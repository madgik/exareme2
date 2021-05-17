import json
import uuid

import pytest
import uuid as uuid

from mipengine.common.node_tasks_DTOs import ColumnInfo, TableSchema
from tests.integration_tests import nodes_communication

local_node = nodes_communication.get_celery_app("localnode1")
local_node_create_table = nodes_communication.get_celery_create_table_signature(
    local_node
)
local_node_get_tables = nodes_communication.get_celery_get_tables_signature(local_node)
local_node_get_table_schema = nodes_communication.get_celery_get_table_schema_signature(
    local_node
)
local_node_get_table_data = nodes_communication.get_celery_get_table_data_signature(
    local_node
)
local_node_create_merge_table = (
    nodes_communication.get_celery_create_merge_table_signature(local_node)
)
local_node_get_merge_tables = nodes_communication.get_celery_get_merge_tables_signature(
    local_node
)
local_node_create_view = nodes_communication.get_celery_create_view_signature(
    local_node
)
local_node_get_views = nodes_communication.get_celery_get_views_signature(local_node)
local_node_create_remote_table = (
    nodes_communication.get_celery_create_remote_table_signature(local_node)
)
local_node_get_remote_tables = (
    nodes_communication.get_celery_get_remote_tables_signature(local_node)
)
local_node_cleanup = nodes_communication.get_celery_cleanup_signature(local_node)

context_id = "HISTOGRAMS"


@pytest.fixture(autouse=True)
def cleanup_tables():
    yield

    local_node_cleanup.delay(context_id=context_id.lower()).get()


def test_sql_injection_get_tables():
    with pytest.raises(ValueError):
        local_node_get_tables.delay(context_id="Robert'); DROP TABLE data; --").get()


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
        local_node_create_view.delay(
            context_id=context_id,
            command_id=str(uuid.uuid1()).replace("-", ""),
            pathology=pathology,
            datasets=datasets,
            columns=columns,
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
        local_node_create_view.delay(
            context_id=context_id,
            command_id=str(uuid.uuid1()).replace("-", ""),
            pathology=pathology,
            datasets=datasets,
            columns=columns,
        ).get()


def test_sql_injection_create_view_filters():
    with pytest.raises(ValueError):
        columns = [
            "dataset",
            "age_value",
            "gcs_motor_response_scale",
            "pupil_reactivity_right_eye_result",
        ]
        datasets = ["edsd"]
        pathology = "tbi"
        filters = {
            "condition": "Robert'); DROP TABLE data; --",
            "rules": [
                {
                    "id": "age_value",
                    "field": "age_value",
                    "type": "int",
                    "input": "number",
                    "operator": "equal",
                    "value": 17,
                }
            ],
            "valid": True,
        }
        local_node_create_view.delay(
            context_id=context_id,
            command_id=str(uuid.uuid1()).replace("-", ""),
            pathology=pathology,
            datasets=datasets,
            columns=columns,
            filters=filters,
        ).get()


def test_sql_injection_get_remote_tables():
    with pytest.raises(ValueError):
        local_node_get_remote_tables.delay(
            context_id="Robert'); DROP TABLE data; --"
        ).get()
