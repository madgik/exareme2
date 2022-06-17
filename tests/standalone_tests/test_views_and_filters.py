import uuid

import pytest

from mipengine.datatypes import DType
from mipengine.node_exceptions import DataModelUnavailable
from mipengine.node_exceptions import DatasetUnavailable
from mipengine.node_exceptions import InsufficientDataError
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema
from tests.standalone_tests import algorithms_url
from tests.standalone_tests.nodes_communication_helper import get_celery_task_signature
from tests.standalone_tests.std_output_logger import StdOutputLogger

TASKS_TIMEOUT = 60


@pytest.fixture
def request_id():
    return "testviews" + uuid.uuid4().hex + "request"


@pytest.fixture
def context_id():
    return "testviews" + uuid.uuid4().hex


def test_view_without_filters(
    request_id,
    context_id,
    load_data_localnode1,
    rabbitmq_localnode1,
    localnode1_celery_app,
    use_localnode1_database,
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
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    )
    table_name = localnode1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    values = [[1, 0.1, "test1"], [2, 0.2, None], [3, 0.3, "test3"]]
    task_signature = get_celery_task_signature("insert_data_to_table")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=table_name,
        values=values,
    )
    localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )

    columns = ["col1", "col3"]
    task_signature = get_celery_task_signature("create_view")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        table_name=table_name,
        columns=columns,
        filters=None,
    )
    view_name = localnode1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    task_signature = get_celery_task_signature("get_views")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
    )
    views = localnode1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
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
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view_name,
    )
    schema_result_json = localnode1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )
    assert view_intended_schema == TableSchema.parse_raw(schema_result_json)

    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view_name,
    )
    view_data_json = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
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
    use_localnode1_database,
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
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    )
    table_name = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )

    values = [[1, 0.1, "test1"], [2, 0.2, None], [3, 0.3, "test3"]]
    task_signature = get_celery_task_signature("insert_data_to_table")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=table_name,
        values=values,
    )
    localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )

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
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        table_name=table_name,
        columns=columns,
        filters=filters,
    )
    view_name = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )

    task_signature = get_celery_task_signature("get_views")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
    )
    views = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
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
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view_name,
    )
    schema_result_json = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )
    assert view_intended_schema == TableSchema.parse_raw(schema_result_json)

    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view_name,
    )
    view_data_json = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
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
    use_localnode1_database,
):
    columns = [
        "dataset",
        "age_value",
        "gcs_motor_response_scale",
        "pupil_reactivity_right_eye_result",
    ]
    data_model = "tbi:0.1"
    task_signature = get_celery_task_signature("create_data_model_views")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        data_model=data_model,
        datasets=[],
        columns_per_view=[columns],
    )
    view_name, *_ = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )
    task_signature = get_celery_task_signature("get_views")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
    )
    views = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
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
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view_name,
    )
    schema_result_json = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )
    assert schema == TableSchema.parse_raw(schema_result_json)

    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view_name,
    )
    view_data_json = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )
    view_data = TableData.parse_raw(view_data_json)
    assert len(view_data.columns) == len(schema.columns)
    assert view_data.name == view_name


def test_data_model_view_with_filters(
    request_id,
    context_id,
    load_data_localnode1,
    rabbitmq_localnode1,
    localnode1_celery_app,
    use_localnode1_database,
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
    task_signature = get_celery_task_signature("create_data_model_views")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        data_model=data_model,
        datasets=[],
        columns_per_view=[columns],
        filters=filters,
        dropna=False,
    )
    view_name, *_ = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )

    task_signature = get_celery_task_signature("get_views")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
    )
    views = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
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
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view_name,
    )
    schema_result_json = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )
    assert schema == TableSchema.parse_raw(schema_result_json)

    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view_name,
    )
    view_data_json = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )
    view_data = TableData.parse_raw(view_data_json)
    assert len(view_data.columns) == len(schema.columns)
    assert view_data.name == view_name


def test_data_model_view_dataset_constraint(
    request_id,
    context_id,
    load_data_localnode1,
    rabbitmq_localnode1,
    localnode1_celery_app,
    use_localnode1_database,
):
    columns = [
        "dataset",
    ]
    data_model = "tbi:0.1"
    task_signature = get_celery_task_signature("create_data_model_views")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        data_model=data_model,
        datasets=["dummy_tbi1"],
        columns_per_view=[columns],
        filters=None,
    )
    view_name, *_ = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )

    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view_name,
    )
    view_data_json = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )

    _, dataset_column = TableData.parse_raw(view_data_json).columns
    assert set(dataset_column.data) == {"dummy_tbi1"}


def test_data_model_view_null_constraints(
    request_id,
    context_id,
    load_data_localnode1,
    rabbitmq_localnode1,
    localnode1_celery_app,
    use_localnode1_database,
):
    columns = [
        "gose_score",
    ]
    data_model = "tbi:0.1"
    datasets = ["dummy_tbi1"]
    task_signature = get_celery_task_signature("create_data_model_views")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        data_model=data_model,
        datasets=datasets,
        columns_per_view=[columns],
        filters=None,
    )
    view_name_without_nulls, *_ = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )

    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view_name_without_nulls,
    )
    view_data_json = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )
    _, gose_score_column = TableData.parse_raw(view_data_json).columns
    assert None not in gose_score_column.data

    task_signature = get_celery_task_signature("create_data_model_views")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        data_model=data_model,
        datasets=datasets,
        columns_per_view=[columns],
        filters=None,
        dropna=False,
    )
    view_name_with_nulls, *_ = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )

    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view_name_with_nulls,
    )
    view_data_json = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )
    _, gose_score_column = TableData.parse_raw(view_data_json).columns

    assert None in gose_score_column.data


def test_data_model_view_min_rows_checks(
    request_id,
    context_id,
    load_data_localnode1,
    rabbitmq_localnode1,
    localnode1_celery_app,
    use_localnode1_database,
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
        task_signature = get_celery_task_signature("create_data_model_views")
        async_result = localnode1_celery_app.queue_task(
            task_signature=task_signature,
            logger=StdOutputLogger(),
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            data_model=data_model,
            datasets=[],
            columns_per_view=[columns],
            filters=filters,
        )
        localnode1_celery_app.get_result(
            async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
        )

        # Check the same view creation with min rows check disabled
        task_signature = get_celery_task_signature("create_data_model_views")
        async_result = localnode1_celery_app.queue_task(
            task_signature=task_signature,
            logger=StdOutputLogger(),
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            data_model=data_model,
            datasets=[],
            columns_per_view=[columns],
            filters=filters,
            check_min_rows=False,
        )
        localnode1_celery_app.get_result(
            async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
        )


def test_data_model_view_with_data_model_unavailable_exception(
    request_id,
    context_id,
    load_data_localnode1,
    rabbitmq_localnode1,
    localnode1_celery_app,
    use_localnode1_database,
):
    columns = [
        "dataset",
    ]
    data_model = "non_existing"
    with pytest.raises(DataModelUnavailable) as exc:
        task_signature = get_celery_task_signature("create_data_model_views")
        async_result = localnode1_celery_app.queue_task(
            task_signature=task_signature,
            logger=StdOutputLogger(),
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            data_model=data_model,
            datasets=[],
            columns_per_view=[columns],
        )
        localnode1_celery_app.get_result(
            async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
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
    use_localnode1_database,
):
    columns = [
        "dataset",
    ]
    data_model = "tbi:0.1"
    with pytest.raises(DatasetUnavailable) as exc:
        task_signature = get_celery_task_signature("create_data_model_views")
        async_result = localnode1_celery_app.queue_task(
            task_signature=task_signature,
            logger=StdOutputLogger(),
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            data_model=data_model,
            datasets=["non_existing"],
            columns_per_view=[columns],
        )
        localnode1_celery_app.get_result(
            async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
        )

    assert (
        f"Dataset 'non_existing' is not available in node: 'testlocalnode1'"
        in exc.value.message
    )


def test_multiple_data_model_views(
    request_id,
    context_id,
    load_data_localnode1,
    rabbitmq_localnode1,
    localnode1_celery_app,
    use_localnode1_database,
):
    columns_per_view = [
        [
            "age_value",
            "gcs_motor_response_scale",
        ],
        [
            "dataset",
            "pupil_reactivity_right_eye_result",
        ],
    ]
    data_model = "tbi:0.1"
    task_signature = get_celery_task_signature("create_data_model_views")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        data_model=data_model,
        datasets=[],
        columns_per_view=columns_per_view,
    )
    view1_name, view2_name = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )

    task_signature = get_celery_task_signature("get_views")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
    )
    views = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )
    assert view1_name in views
    assert view2_name in views
    schema1 = TableSchema(
        columns=[
            ColumnInfo(name="row_id", dtype=DType.INT),
            ColumnInfo(name="age_value", dtype=DType.INT),
            ColumnInfo(name="gcs_motor_response_scale", dtype=DType.STR),
        ]
    )
    task_signature = get_celery_task_signature("get_table_schema")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view1_name,
    )
    schema_result_json = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )
    assert schema1 == TableSchema.parse_raw(schema_result_json)

    schema2 = TableSchema(
        columns=[
            ColumnInfo(name="row_id", dtype=DType.INT),
            ColumnInfo(name="dataset", dtype=DType.STR),
            ColumnInfo(name="pupil_reactivity_right_eye_result", dtype=DType.STR),
        ]
    )
    task_signature = get_celery_task_signature("get_table_schema")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view2_name,
    )
    schema_result_json = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )
    assert schema2 == TableSchema.parse_raw(schema_result_json)


def test_multiple_data_model_views_null_constraints(
    request_id,
    context_id,
    load_data_localnode1,
    rabbitmq_localnode1,
    localnode1_celery_app,
    use_localnode1_database,
):
    columns_per_view = [
        [
            "gose_score",  # Column with values
        ],
        [
            "gcs_eye_response_scale",  # Column without values
        ],
    ]
    data_model = "tbi:0.1"
    datasets = ["dummy_tbi1"]

    task_signature = get_celery_task_signature("create_data_model_views")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        data_model=data_model,
        datasets=datasets,
        columns_per_view=columns_per_view,
        filters=None,
        check_min_rows=False,
    )
    view_name_with_values, view_name_with_nulls = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )

    # Check that the all null view doesn't have any rows (All rows were dropped)
    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view_name_with_nulls,
    )
    view_name_with_nulls_data_json = localnode1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    _, gose_score_column = TableData.parse_raw(view_name_with_nulls_data_json).columns
    assert len(gose_score_column.data) == 0

    # Check that the view that didn't have nulls is also empty due to multiple views having linked null constraints
    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view_name_with_values,
    )
    view_name_with_values_data_json = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )
    _, gcs_eye_response_scale_column = TableData.parse_raw(
        view_name_with_values_data_json
    ).columns
    assert len(gcs_eye_response_scale_column.data) == 0
