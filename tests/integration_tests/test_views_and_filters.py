import uuid

import pytest

from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema
from tests.integration_tests.nodes_communication import get_celery_task_signature
from tests.integration_tests.nodes_communication import get_celery_app

local_node_id = "localnode1"
local_node = get_celery_app(local_node_id)
local_node_create_table = get_celery_task_signature(local_node, "create_table")
local_node_insert_data_to_table = get_celery_task_signature(
    local_node, "insert_data_to_table"
)
local_node_create_pathology_view = get_celery_task_signature(
    local_node, "create_pathology_view"
)
local_node_create_view = get_celery_task_signature(local_node, "create_view")
local_node_get_views = get_celery_task_signature(local_node, "get_views")
local_node_get_view_data = get_celery_task_signature(local_node, "get_table_data")
local_node_get_view_schema = get_celery_task_signature(local_node, "get_table_schema")
local_node_cleanup = get_celery_task_signature(local_node, "clean_up")


@pytest.fixture(autouse=True)
def context_id():
    context_id = "test_views_" + str(uuid.uuid4()).replace("-", "")

    yield context_id

    local_node_cleanup.delay(context_id=context_id.lower()).get()


def test_create_view_and_get_view_without_filters(context_id):
    table_schema = TableSchema(
        [
            ColumnInfo("col1", "int"),
            ColumnInfo("col2", "real"),
            ColumnInfo("col3", "text"),
        ]
    )

    table_name = local_node_create_table.delay(
        context_id=context_id,
        command_id=str(uuid.uuid4()).replace("-", ""),
        schema_json=table_schema.to_json(),
    ).get()

    values = [[1, 0.1, "test1"], [2, 0.2, None], [3, 0.3, "test3"]]
    local_node_insert_data_to_table.delay(table_name=table_name, values=values).get()
    columns = ["col1", "col3"]
    view_name = local_node_create_view.delay(
        context_id=context_id,
        command_id=str(uuid.uuid1()).replace("-", ""),
        table_name=table_name,
        columns=columns,
        filters=None,
    ).get()

    views = local_node_get_views.delay(context_id=context_id).get()
    assert view_name in views
    view_intended_schema = TableSchema(
        [
            ColumnInfo("col1", "int"),
            ColumnInfo("col3", "text"),
        ]
    )
    schema_result_json = local_node_get_view_schema.delay(table_name=view_name).get()
    assert view_intended_schema == TableSchema.from_json(schema_result_json)

    view_data_json = local_node_get_view_data.delay(table_name=view_name).get()
    view_data = TableData.from_json(view_data_json)
    assert all(
        len(columns) == len(view_intended_schema.columns) for columns in view_data.data
    )
    assert view_data.schema == view_intended_schema


def test_create_view_and_get_view_with_filters(context_id):
    table_schema = TableSchema(
        [
            ColumnInfo("col1", "int"),
            ColumnInfo("col2", "real"),
            ColumnInfo("col3", "text"),
        ]
    )

    table_name = local_node_create_table.delay(
        context_id=context_id,
        command_id=str(uuid.uuid4()).replace("-", ""),
        schema_json=table_schema.to_json(),
    ).get()

    values = [[1, 0.1, "test1"], [2, 0.2, None], [3, 0.3, "test3"]]
    local_node_insert_data_to_table.delay(table_name=table_name, values=values).get()
    columns = ["col1", "col3"]
    rules = {
        "condition": "AND",
        "rules": [
            {
                "condition": "OR",
                "rules": [
                    {
                        "id": "col1",
                        "field": "col1",
                        "type": "int",
                        "input": "number",
                        "operator": "equal",
                        "value": 3,
                    }
                ],
            }
        ],
        "valid": True,
    }
    view_name = local_node_create_view.delay(
        context_id=context_id,
        command_id=str(uuid.uuid1()).replace("-", ""),
        table_name=table_name,
        columns=columns,
        filters=rules,
    ).get()

    views = local_node_get_views.delay(context_id=context_id).get()
    assert view_name in views
    view_intended_schema = TableSchema(
        [
            ColumnInfo("col1", "int"),
            ColumnInfo("col3", "text"),
        ]
    )
    schema_result_json = local_node_get_view_schema.delay(table_name=view_name).get()
    assert view_intended_schema == TableSchema.from_json(schema_result_json)

    view_data_json = local_node_get_view_data.delay(table_name=view_name).get()
    view_data = TableData.from_json(view_data_json)
    assert len(view_data.data) == 1
    assert all(
        len(columns) == len(view_intended_schema.columns) for columns in view_data.data
    )
    assert view_data.schema == view_intended_schema


def test_create_pathology_view_and_get_view_without_filters(context_id):
    columns = [
        "dataset",
        "age_value",
        "gcs_motor_response_scale",
        "pupil_reactivity_right_eye_result",
    ]
    pathology = "tbi"
    view_name = local_node_create_pathology_view.delay(
        context_id=context_id,
        command_id=str(uuid.uuid1()).replace("-", ""),
        pathology=pathology,
        columns=columns,
        filters=None,
    ).get()
    views = local_node_get_views.delay(context_id=context_id).get()
    assert view_name in views

    schema = TableSchema(
        [
            ColumnInfo("row_id", "int"),
            ColumnInfo("dataset", "text"),
            ColumnInfo("age_value", "int"),
            ColumnInfo("gcs_motor_response_scale", "text"),
            ColumnInfo("pupil_reactivity_right_eye_result", "text"),
        ]
    )
    schema_result_json = local_node_get_view_schema.delay(table_name=view_name).get()
    assert schema == TableSchema.from_json(schema_result_json)

    view_data_json = local_node_get_view_data.delay(table_name=view_name).get()
    view_data = TableData.from_json(view_data_json)
    assert all(len(columns) == len(schema.columns) for columns in view_data.data)
    assert view_data.schema == schema

    view_schema_json = local_node_get_view_schema.delay(table_name=view_name).get()
    view_schema = TableSchema.from_json(view_schema_json)
    assert view_schema == schema


def test_create_pathology_view_and_get_view_with_filters(context_id):
    columns = [
        "dataset",
        "age_value",
        "gcs_motor_response_scale",
        "pupil_reactivity_right_eye_result",
    ]
    pathology = "tbi"
    rules = {
        "condition": "AND",
        "rules": [
            {
                "condition": "OR",
                "rules": [
                    {
                        "id": "age_value",
                        "field": "age_value",
                        "type": "int",
                        "input": "number",
                        "operator": "greater",
                        "value": 30,
                    }
                ],
            }
        ],
        "valid": True,
    }
    view_name = local_node_create_pathology_view.delay(
        context_id=context_id,
        command_id=str(uuid.uuid1()).replace("-", ""),
        pathology=pathology,
        columns=columns,
        filters=rules,
    ).get()
    views = local_node_get_views.delay(context_id=context_id).get()
    assert view_name in views

    schema = TableSchema(
        [
            ColumnInfo("row_id", "int"),
            ColumnInfo("dataset", "text"),
            ColumnInfo("age_value", "int"),
            ColumnInfo("gcs_motor_response_scale", "text"),
            ColumnInfo("pupil_reactivity_right_eye_result", "text"),
        ]
    )
    schema_result_json = local_node_get_view_schema.delay(table_name=view_name).get()
    assert schema == TableSchema.from_json(schema_result_json)

    view_data_json = local_node_get_view_data.delay(table_name=view_name).get()
    view_data = TableData.from_json(view_data_json)
    assert all(len(columns) == len(schema.columns) for columns in view_data.data)
    assert view_data.schema == schema

    view_schema_json = local_node_get_view_schema.delay(table_name=view_name).get()
    view_schema = TableSchema.from_json(view_schema_json)
    assert view_schema == schema
