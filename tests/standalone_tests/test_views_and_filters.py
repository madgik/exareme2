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

TASKS_TIMEOUT = 60


@pytest.fixture(autouse=True)
def request_id():
    return "testviews" + uuid.uuid4().hex + "request"


@pytest.fixture(autouse=True)
def context_id(request_id, localnode1_celery_app):
    context_id = "testviews" + uuid.uuid4().hex

    yield context_id

    task_signature = get_celery_task_signature("clean_up")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        request_id=request_id,
        context_id=context_id.lower(),
    )
    localnode1_celery_app.get_result(async_result=async_result, timeout=TASKS_TIMEOUT)


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

    task_signature = get_celery_task_signature("create_table")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    )
    table_name = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
    )

    values = [[1, 0.1, "test1"], [2, 0.2, None], [3, 0.3, "test3"]]
    task_signature = get_celery_task_signature("insert_data_to_table")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        request_id=request_id,
        table_name=table_name,
        values=values,
    )
    localnode1_celery_app.get_result(async_result=async_result, timeout=TASKS_TIMEOUT)

    columns = ["col1", "col3"]
    task_signature = get_celery_task_signature("create_view")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        table_name=table_name,
        columns=columns,
        filters=None,
    )
    view_name = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
    )

    task_signature = get_celery_task_signature("get_views")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature, request_id=request_id, context_id=context_id
    )
    views = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
    )
    assert view_name in views
    view_intended_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )
    task_signature = get_celery_task_signature("get_table_schema")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature, request_id=request_id, table_name=view_name
    )
    schema_result_json = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
    )
    assert view_intended_schema == TableSchema.parse_raw(schema_result_json)

    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature, request_id=request_id, table_name=view_name
    )
    view_data_json = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
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

    task_signature = get_celery_task_signature("create_table")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    )
    table_name = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
    )

    values = [[1, 0.1, "test1"], [2, 0.2, None], [3, 0.3, "test3"]]
    task_signature = get_celery_task_signature("insert_data_to_table")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        request_id=request_id,
        table_name=table_name,
        values=values,
    )
    localnode1_celery_app.get_result(async_result=async_result, timeout=TASKS_TIMEOUT)
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

    task_signature = get_celery_task_signature("create_view")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        table_name=table_name,
        columns=columns,
        filters=filters,
    )
    view_name = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
    )

    task_signature = get_celery_task_signature("get_views")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature, request_id=request_id, context_id=context_id
    )
    views = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
    )
    assert view_name in views
    view_intended_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )

    task_signature = get_celery_task_signature("get_table_schema")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature, request_id=request_id, table_name=view_name
    )
    schema_result_json = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
    )
    assert view_intended_schema == TableSchema.parse_raw(schema_result_json)

    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature, request_id=request_id, table_name=view_name
    )
    view_data_json = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
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
    task_signature = get_celery_task_signature("create_data_model_view")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        data_model=data_model,
        datasets=[],
        columns=columns,
    )
    view_name = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
    )
    task_signature = get_celery_task_signature("get_views")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature, request_id=request_id, context_id=context_id
    )
    views = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
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
    task_signature = get_celery_task_signature("get_table_schema")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature, request_id=request_id, table_name=view_name
    )
    schema_result_json = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
    )
    assert schema == TableSchema.parse_raw(schema_result_json)

    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature, request_id=request_id, table_name=view_name
    )
    view_data_json = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
    )
    view_data = TableData.parse_raw(view_data_json)
    assert len(view_data.columns) == len(schema.columns)
    assert view_data.name == view_name

    task_signature = get_celery_task_signature("get_table_schema")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature, request_id=request_id, table_name=view_name
    )
    view_schema_json = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
    )
    view_schema = TableSchema.parse_raw(view_schema_json)
    assert view_schema == schema


@pytest.mark.skip(reason="fails because of dropna")
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
    task_signature = get_celery_task_signature("create_data_model_view")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        data_model=data_model,
        datasets=[],
        columns=columns,
        filters=filters,
        dropna=False,
    )
    view_name = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
    )

    task_signature = get_celery_task_signature("get_views")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature, request_id=request_id, context_id=context_id
    )
    views = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
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

    task_signature = get_celery_task_signature("get_table_schema")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature, request_id=request_id, table_name=view_name
    )
    schema_result_json = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
    )
    assert schema == TableSchema.parse_raw(schema_result_json)

    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature, request_id=request_id, table_name=view_name
    )
    view_data_json = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
    )
    view_data = TableData.parse_raw(view_data_json)
    assert len(view_data.columns) == len(schema.columns)
    assert view_data.name == view_name

    task_signature = get_celery_task_signature("get_table_schema")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature, request_id=request_id, table_name=view_name
    )
    view_schema_json = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
    )
    view_schema = TableSchema.parse_raw(view_schema_json)
    assert view_schema == schema


@pytest.mark.skip(reason="fails, needs fixing")
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

    task_signature = get_celery_task_signature("create_data_model_view")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        data_model=data_model,
        datasets=["dummy_tbi1"],
        columns=columns,
        filters=None,
    )
    view_name = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
    )

    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature, request_id=request_id, table_name=view_name
    )
    view_data_json = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
    )
    _, dataset_column = TableData.parse_raw(view_data_json).columns
    assert set(dataset_column.data) == {"dummy_tbi1"}


@pytest.mark.skip(reason="fails, needs fixing")
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

    task_signature = get_celery_task_signature("create_data_model_view")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        data_model=data_model,
        datasets=datasets,
        columns=columns,
        filters=None,
    )
    view_name_without_nulls = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
    )

    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        request_id=request_id,
        table_name=view_name_without_nulls,
    )
    view_data_json = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
    )
    _, gose_score_column = TableData.parse_raw(view_data_json).columns
    assert None not in gose_score_column.data

    task_signature = get_celery_task_signature("create_data_model_view")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        data_model=data_model,
        datasets=datasets,
        columns=columns,
        filters=None,
        dropna=False,
    )
    view_name_with_nulls = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
    )

    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        request_id=request_id,
        table_name=view_name_with_nulls,
    )
    view_data_json = localnode1_celery_app.get_result(
        async_result=async_result, timeout=TASKS_TIMEOUT
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
        task_signature = get_celery_task_signature("create_data_model_view")
        async_result = localnode1_celery_app.queue_task(
            task_signature=task_signature,
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            data_model=data_model,
            datasets=[],
            columns=columns,
            filters=filters,
        )
        localnode1_celery_app.get_result(
            async_result=async_result, timeout=TASKS_TIMEOUT
        )


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
        task_signature = get_celery_task_signature("create_data_model_view")
        async_result = localnode1_celery_app.queue_task(
            task_signature=task_signature,
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            data_model=data_model,
            datasets=[],
            columns=columns,
        )
        localnode1_celery_app.get_result(
            async_result=async_result, timeout=TASKS_TIMEOUT
        )

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
        task_signature = get_celery_task_signature("create_data_model_view")
        async_result = localnode1_celery_app.queue_task(
            task_signature=task_signature,
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            data_model=data_model,
            datasets=["non_existing"],
            columns=columns,
        )
        localnode1_celery_app.get_result(
            async_result=async_result, timeout=TASKS_TIMEOUT
        )

    assert (
        f"Dataset 'non_existing' is not available in node: 'testlocalnode1'"
        in exc.value.message
    )
