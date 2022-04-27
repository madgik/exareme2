import json
import uuid

import pytest
import requests

from mipengine.datatypes import DType
from mipengine.node_exceptions import DataModelUnavailable
from mipengine.node_exceptions import DatasetUnavailable
from mipengine.node_exceptions import InsufficientDataError
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema
from tests.standalone_tests import algorithms_url
from tests.standalone_tests.nodes_communication_helper import get_celery_task_signature


@pytest.fixture(autouse=True)
def request_id():
    return "testviews" + uuid.uuid4().hex + "request"


@pytest.fixture(autouse=True)
def context_id(request_id, localnode1_celery_app):
    context_id = "testviews" + uuid.uuid4().hex

    yield context_id

    get_celery_task_signature(localnode1_celery_app, "clean_up").delay(
        request_id=request_id, context_id=context_id.lower()
    ).get()


def test_view_without_filters(
    request_id,
    context_id,
    load_data_localnode1,
    rabbitmq_localnode1,
    localnode1_celery_app,
):
    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )

    table_name = (
        get_celery_task_signature(localnode1_celery_app, "create_table")
        .delay(
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            schema_json=table_schema.json(),
        )
        .get()
    )

    values = [[1, 0.1, "test1"], [2, 0.2, None], [3, 0.3, "test3"]]
    get_celery_task_signature(localnode1_celery_app, "insert_data_to_table").delay(
        request_id=request_id, table_name=table_name, values=values
    ).get()
    columns = ["col1", "col3"]
    view_name = (
        get_celery_task_signature(localnode1_celery_app, "create_view")
        .delay(
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            table_name=table_name,
            columns=columns,
            filters=None,
        )
        .get()
    )

    views = (
        get_celery_task_signature(localnode1_celery_app, "get_views")
        .delay(request_id=request_id, context_id=context_id)
        .get()
    )
    assert view_name in views
    view_intended_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )
    schema_result_json = (
        get_celery_task_signature(localnode1_celery_app, "get_table_schema")
        .delay(request_id=request_id, table_name=view_name)
        .get()
    )
    assert view_intended_schema == TableSchema.parse_raw(schema_result_json)

    view_data_json = (
        get_celery_task_signature(localnode1_celery_app, "get_table_data")
        .delay(request_id=request_id, table_name=view_name)
        .get()
    )
    view_data = TableData.parse_raw(view_data_json)
    assert len(view_data.columns) == len(view_intended_schema.columns)
    assert view_data.name == view_name


def test_view_with_filters(
    request_id,
    context_id,
    load_data_localnode1,
    rabbitmq_localnode1,
    localnode1_celery_app,
):
    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )

    table_name = (
        get_celery_task_signature(localnode1_celery_app, "create_table")
        .delay(
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            schema_json=table_schema.json(),
        )
        .get()
    )

    values = [[1, 0.1, "test1"], [2, 0.2, None], [3, 0.3, "test3"]]
    get_celery_task_signature(localnode1_celery_app, "insert_data_to_table").delay(
        request_id=request_id, table_name=table_name, values=values
    ).get()
    columns = ["col1", "col3"]
    filters = {
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
    view_name = (
        get_celery_task_signature(localnode1_celery_app, "create_view")
        .delay(
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            table_name=table_name,
            columns=columns,
            filters=filters,
        )
        .get()
    )

    views = (
        get_celery_task_signature(localnode1_celery_app, "get_views")
        .delay(request_id=request_id, context_id=context_id)
        .get()
    )
    assert view_name in views
    view_intended_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )
    schema_result_json = (
        get_celery_task_signature(localnode1_celery_app, "get_table_schema")
        .delay(request_id=request_id, table_name=view_name)
        .get()
    )
    assert view_intended_schema == TableSchema.parse_raw(schema_result_json)

    view_data_json = (
        get_celery_task_signature(localnode1_celery_app, "get_table_data")
        .delay(request_id=request_id, table_name=view_name)
        .get()
    )
    view_data = TableData.parse_raw(view_data_json)
    assert len(view_data.columns) == 2
    assert len(view_data.columns) == len(view_intended_schema.columns)
    assert view_data.name == view_name


def test_data_model_view_without_filters(
    request_id,
    context_id,
    load_data_localnode1,
    rabbitmq_localnode1,
    localnode1_celery_app,
):
    columns = [
        "dataset",
        "age_value",
        "gcs_motor_response_scale",
        "pupil_reactivity_right_eye_result",
    ]
    data_model = "tbi:0.1"
    view_name = (
        get_celery_task_signature(localnode1_celery_app, "create_data_model_view")
        .delay(
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            data_model=data_model,
            datasets=[],
            columns=columns,
        )
        .get()
    )
    views = (
        get_celery_task_signature(localnode1_celery_app, "get_views")
        .delay(request_id=request_id, context_id=context_id)
        .get()
    )
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
    schema_result_json = (
        get_celery_task_signature(localnode1_celery_app, "get_table_schema")
        .delay(request_id=request_id, table_name=view_name)
        .get()
    )
    assert schema == TableSchema.parse_raw(schema_result_json)

    view_data_json = (
        get_celery_task_signature(localnode1_celery_app, "get_table_data")
        .delay(request_id=request_id, table_name=view_name)
        .get()
    )
    view_data = TableData.parse_raw(view_data_json)
    assert len(view_data.columns) == len(schema.columns)
    assert view_data.name == view_name

    view_schema_json = (
        get_celery_task_signature(localnode1_celery_app, "get_table_schema")
        .delay(request_id=request_id, table_name=view_name)
        .get()
    )
    view_schema = TableSchema.parse_raw(view_schema_json)
    assert view_schema == schema


def test_data_model_view_with_filters(
    request_id,
    context_id,
    load_data_localnode1,
    rabbitmq_localnode1,
    localnode1_celery_app,
):
    columns = [
        "dataset",
        "age_value",
        "gcs_motor_response_scale",
        "pupil_reactivity_right_eye_result",
    ]
    data_model = "tbi:0.1"
    filters = {
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
    view_name = (
        get_celery_task_signature(localnode1_celery_app, "create_data_model_view")
        .delay(
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            data_model=data_model,
            datasets=[],
            columns=columns,
            filters=filters,
            drop_na=False,
        )
        .get()
    )
    views = (
        get_celery_task_signature(localnode1_celery_app, "get_views")
        .delay(request_id=request_id, context_id=context_id)
        .get()
    )
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
    schema_result_json = (
        get_celery_task_signature(localnode1_celery_app, "get_table_schema")
        .delay(request_id=request_id, table_name=view_name)
        .get()
    )
    assert schema == TableSchema.parse_raw(schema_result_json)

    view_data_json = (
        get_celery_task_signature(localnode1_celery_app, "get_table_data")
        .delay(request_id=request_id, table_name=view_name)
        .get()
    )
    view_data = TableData.parse_raw(view_data_json)
    assert len(view_data.columns) == len(schema.columns)
    assert view_data.name == view_name

    view_schema_json = (
        get_celery_task_signature(localnode1_celery_app, "get_table_schema")
        .delay(request_id=request_id, table_name=view_name)
        .get()
    )
    view_schema = TableSchema.parse_raw(view_schema_json)
    assert view_schema == schema


def test_data_model_view_dataset_constraint(
    request_id,
    context_id,
    load_data_localnode1,
    rabbitmq_localnode1,
    localnode1_celery_app,
):
    columns = [
        "dataset",
    ]
    data_model = "tbi:0.1"
    view_name = (
        get_celery_task_signature(localnode1_celery_app, "create_data_model_view")
        .delay(
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            data_model=data_model,
            datasets=["dummy_tbi1"],
            columns=columns,
            filters=None,
        )
        .get()
    )

    view_data_json = (
        get_celery_task_signature(localnode1_celery_app, "get_table_data")
        .delay(request_id=request_id, table_name=view_name)
        .get()
    )

    _, dataset_column = TableData.parse_raw(view_data_json).columns
    assert set(dataset_column.data) == {"dummy_tbi1"}


def test_data_model_view_null_constraints(
    request_id,
    context_id,
    load_data_localnode1,
    rabbitmq_localnode1,
    localnode1_celery_app,
):
    columns = [
        "gose_score",
    ]
    data_model = "tbi:0.1"
    datasets = ["dummy_tbi1"]
    view_name_without_nulls = (
        get_celery_task_signature(localnode1_celery_app, "create_data_model_view")
        .delay(
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            data_model=data_model,
            datasets=datasets,
            columns=columns,
            filters=None,
        )
        .get()
    )

    view_data_json = (
        get_celery_task_signature(localnode1_celery_app, "get_table_data")
        .delay(request_id=request_id, table_name=view_name_without_nulls)
        .get()
    )

    _, gose_score_column = TableData.parse_raw(view_data_json).columns
    assert None not in gose_score_column.data

    view_name_with_nulls = (
        get_celery_task_signature(localnode1_celery_app, "create_data_model_view")
        .delay(
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            data_model=data_model,
            datasets=datasets,
            columns=columns,
            filters=None,
            drop_na=False,
        )
        .get()
    )

    view_data_json = (
        get_celery_task_signature(localnode1_celery_app, "get_table_data")
        .delay(request_id=request_id, table_name=view_name_with_nulls)
        .get()
    )
    _, gose_score_column = TableData.parse_raw(view_data_json).columns

    assert None in gose_score_column.data


def test_data_model_view_with_privacy_error(
    request_id,
    context_id,
    load_data_localnode1,
    rabbitmq_localnode1,
    localnode1_celery_app,
):
    columns = [
        "dataset",
        "age_value",
        "gcs_motor_response_scale",
        "pupil_reactivity_right_eye_result",
    ]
    data_model = "tbi:0.1"

    # Adding a filter that cannot be matched
    filters = {
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
        get_celery_task_signature(
            localnode1_celery_app, "create_data_model_view"
        ).delay(
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            data_model=data_model,
            datasets=[],
            columns=columns,
            filters=filters,
        ).get()


def test_data_model_view_with_data_model_unavailable_exception(
    request_id,
    context_id,
    load_data_localnode1,
    rabbitmq_localnode1,
    localnode1_celery_app,
):
    columns = [
        "dataset",
    ]
    data_model = "non_existing"
    with pytest.raises(DataModelUnavailable) as exc:
        get_celery_task_signature(
            localnode1_celery_app, "create_data_model_view"
        ).delay(
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            data_model=data_model,
            datasets=[],
            columns=columns,
        ).get()

    assert (
        f"Data model 'non_existing' is not available in node: 'testlocalnode1'"
        in exc.value.message
    )


def test_data_model_view_with_dataset_unavailable_exception(
    request_id,
    context_id,
    load_data_localnode1,
    rabbitmq_localnode1,
    localnode1_celery_app,
):
    columns = [
        "dataset",
    ]
    data_model = "tbi:0.1"
    with pytest.raises(DatasetUnavailable) as exc:
        get_celery_task_signature(
            localnode1_celery_app, "create_data_model_view"
        ).delay(
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            data_model=data_model,
            datasets=["non_existing"],
            columns=columns,
        ).get()

    assert (
        f"Dataset 'non_existing' is not available in node: 'testlocalnode1'"
        in exc.value.message
    )
