import json
import uuid

import pytest
import requests

from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.datatypes import DType
from mipengine.node_tasks_DTOs import InsufficientDataError
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema
from tests.dev_env_tests import algorithms_url
from tests.dev_env_tests.nodes_communication import get_celery_task_signature
from tests.dev_env_tests.nodes_communication import get_celery_app

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
def request_id():
    return "test_views_" + uuid.uuid4().hex + "_request"


@pytest.fixture(autouse=True)
def context_id(request_id):
    context_id = "test_views_" + uuid.uuid4().hex

    yield context_id

    local_node_cleanup.delay(request_id=request_id, context_id=context_id.lower()).get()


def test_view_without_filters(request_id, context_id):
    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )

    table_name = local_node_create_table.delay(
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    ).get()

    values = [[1, 0.1, "test1"], [2, 0.2, None], [3, 0.3, "test3"]]
    local_node_insert_data_to_table.delay(
        request_id=request_id, table_name=table_name, values=values
    ).get()
    columns = ["col1", "col3"]
    view_name = local_node_create_view.delay(
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        table_name=table_name,
        columns=columns,
        filters=None,
    ).get()

    views = local_node_get_views.delay(
        request_id=request_id, context_id=context_id
    ).get()
    assert view_name in views
    view_intended_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )
    schema_result_json = local_node_get_view_schema.delay(
        request_id=request_id, table_name=view_name
    ).get()
    assert view_intended_schema == TableSchema.parse_raw(schema_result_json)

    view_data_json = local_node_get_view_data.delay(
        request_id=request_id, table_name=view_name
    ).get()
    view_data = TableData.parse_raw(view_data_json)
    assert len(view_data.columns) == len(view_intended_schema.columns)
    assert view_data.name == view_name


def test_view_with_filters(request_id, context_id):
    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )

    table_name = local_node_create_table.delay(
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    ).get()

    values = [[1, 0.1, "test1"], [2, 0.2, None], [3, 0.3, "test3"]]
    local_node_insert_data_to_table.delay(
        request_id=request_id, table_name=table_name, values=values
    ).get()
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
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        table_name=table_name,
        columns=columns,
        filters=rules,
    ).get()

    views = local_node_get_views.delay(
        request_id=request_id, context_id=context_id
    ).get()
    assert view_name in views
    view_intended_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )
    schema_result_json = local_node_get_view_schema.delay(
        request_id=request_id, table_name=view_name
    ).get()
    assert view_intended_schema == TableSchema.parse_raw(schema_result_json)

    view_data_json = local_node_get_view_data.delay(
        request_id=request_id, table_name=view_name
    ).get()
    view_data = TableData.parse_raw(view_data_json)
    assert len(view_data.columns) == 2
    assert len(view_data.columns) == len(view_intended_schema.columns)
    assert view_data.name == view_name


def test_pathology_view_without_filters(request_id, context_id):
    columns = [
        "dataset",
        "age_value",
        "gcs_motor_response_scale",
        "pupil_reactivity_right_eye_result",
    ]
    pathology = "tbi"
    view_name = local_node_create_pathology_view.delay(
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        pathology=pathology,
        columns=columns,
        filters=None,
    ).get()
    views = local_node_get_views.delay(
        request_id=request_id, context_id=context_id
    ).get()
    assert view_name in views

    schema = TableSchema(
        columns=[
            ColumnInfo(name="row_id", dtype=DType.INT),
            ColumnInfo(name="dataset", dtype=DType.STR),
            ColumnInfo(name="age_value", dtype=DType.INT),
            ColumnInfo(name="gcs_motor_response_scale", dtype=DType.STR),
            ColumnInfo(name="pupil_reactivity_right_eye_result", dtype=DType.STR),
        ]
    )
    schema_result_json = local_node_get_view_schema.delay(
        request_id=request_id, table_name=view_name
    ).get()
    print(TableSchema.parse_raw(schema_result_json))
    assert schema == TableSchema.parse_raw(schema_result_json)

    view_data_json = local_node_get_view_data.delay(
        request_id=request_id, table_name=view_name
    ).get()
    view_data = TableData.parse_raw(view_data_json)
    assert len(view_data.columns) == len(schema.columns)
    assert view_data.name == view_name

    view_schema_json = local_node_get_view_schema.delay(
        request_id=request_id, table_name=view_name
    ).get()
    view_schema = TableSchema.parse_raw(view_schema_json)
    assert view_schema == schema


def test_pathology_view_with_filters(request_id, context_id):
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
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        pathology=pathology,
        columns=columns,
        filters=rules,
    ).get()
    views = local_node_get_views.delay(
        request_id=request_id, context_id=context_id
    ).get()
    assert view_name in views

    schema = TableSchema(
        columns=[
            ColumnInfo(name="row_id", dtype=DType.INT),
            ColumnInfo(name="dataset", dtype=DType.STR),
            ColumnInfo(name="age_value", dtype=DType.INT),
            ColumnInfo(name="gcs_motor_response_scale", dtype=DType.STR),
            ColumnInfo(name="pupil_reactivity_right_eye_result", dtype=DType.STR),
        ]
    )
    schema_result_json = local_node_get_view_schema.delay(
        request_id=request_id, table_name=view_name
    ).get()
    assert schema == TableSchema.parse_raw(schema_result_json)

    view_data_json = local_node_get_view_data.delay(
        request_id=request_id, table_name=view_name
    ).get()
    view_data = TableData.parse_raw(view_data_json)
    assert len(view_data.columns) == len(schema.columns)
    assert view_data.name == view_name

    view_schema_json = local_node_get_view_schema.delay(
        request_id=request_id, table_name=view_name
    ).get()
    view_schema = TableSchema.parse_raw(view_schema_json)
    assert view_schema == schema


def test_bad_filters_exception():
    algorithm_name = "standard_deviation"
    request_params = {
        "inputdata": {
            "pathology": "dementia",
            "datasets": ["edsd"],
            "x": [
                "lefthippocampus",
            ],
            "filters": {"whateveeeeeer": "!!!"},
        },
    }

    algorithm_url = algorithms_url + "/" + algorithm_name
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        algorithm_url,
        data=json.dumps(request_params),
        headers=headers,
    )

    assert "Invalid filters format." in response.text
    assert response.status_code == 400


def test_pathology_view_with_privacy_error(request_id, context_id):
    columns = [
        "dataset",
        "age_value",
        "gcs_motor_response_scale",
        "pupil_reactivity_right_eye_result",
    ]
    pathology = "tbi"

    # Adding a filter that cannot be matched
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
                        "value": 200,
                    }
                ],
            }
        ],
        "valid": True,
    }
    with pytest.raises(InsufficientDataError):
        local_node_create_pathology_view.delay(
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            pathology=pathology,
            columns=columns,
            filters=rules,
        ).get()
