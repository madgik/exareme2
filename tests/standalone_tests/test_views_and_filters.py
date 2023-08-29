import json
import uuid

import pytest
import requests

from exareme2.datatypes import DType
from exareme2.exceptions import DataModelUnavailable
from exareme2.exceptions import DatasetUnavailable
from exareme2.exceptions import InsufficientDataError
from exareme2.node_tasks_DTOs import ColumnInfo
from exareme2.node_tasks_DTOs import TableData
from exareme2.node_tasks_DTOs import TableInfo
from exareme2.node_tasks_DTOs import TableSchema
from tests.standalone_tests.conftest import ALGORITHMS_URL
from tests.standalone_tests.conftest import MONETDB_LOCALNODE1_PORT
from tests.standalone_tests.conftest import insert_data_to_localnode
from tests.standalone_tests.nodes_communication_helper import get_celery_task_signature
from tests.standalone_tests.std_output_logger import StdOutputLogger
from tests.standalone_tests.test_smpc_node_tasks import TASKS_TIMEOUT


@pytest.fixture
def request_id():
    return "testviews" + uuid.uuid4().hex + "request"


@pytest.fixture
def context_id():
    return "testviews" + uuid.uuid4().hex


@pytest.fixture
def zero_rows_data_model_view_generating_params():
    data_model = "dementia:0.1"
    columns = [
        "dataset",
    ]
    filters = {
        "condition": "AND",
        "rules": [
            {
                "id": "brainstem",
                "type": "float",
                "input": "number",
                "operator": "greater",
                "value": 99999999,
            }
        ],
        "valid": True,
    }
    return {"data_model": data_model, "columns": columns, "filters": filters}


@pytest.fixture
def five_rows_data_model_view_generating_params():
    data_model = "dementia:0.1"
    columns = [
        "dataset",
        "brainstem",
    ]
    filters = {
        "condition": "AND",
        "rules": [
            {
                "id": "brainstem",
                "type": "float",
                "input": "number",
                "operator": "between",
                "value": [17, 19],
            },
            {
                "id": "dataset",
                "type": "string",
                "input": "str",
                "operator": "equal",
                "value": "edsd0",
            },
        ],
        "valid": True,
    }
    return {"data_model": data_model, "columns": columns, "filters": filters}


@pytest.mark.slow
def test_view_without_filters(
    request_id,
    context_id,
    load_data_localnode1,
    localnode1_node_service,
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
    table_info = TableInfo.parse_raw(
        localnode1_celery_app.get_result(
            async_result=async_result,
            logger=StdOutputLogger(),
            timeout=TASKS_TIMEOUT,
        )
    )

    values = [[1, 0.1, "test1"], [2, 0.2, None], [3, 0.3, "test3"]]
    # task_signature = get_celery_task_signature("insert_data_to_table")
    # async_result = localnode1_celery_app.queue_task(
    #     task_signature=task_signature,
    #     logger=StdOutputLogger(),
    #     request_id=request_id,
    #     table_name=table_info.name,
    #     values=values,
    # )
    # localnode1_celery_app.get_result(
    #     async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    # )
    insert_data_to_localnode(table_info.name, values, MONETDB_LOCALNODE1_PORT)

    columns = ["col1", "col3"]
    task_signature = get_celery_task_signature("create_view")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        table_name=table_info.name,
        columns=columns,
        filters=None,
    )
    view_info = TableInfo.parse_raw(
        localnode1_celery_app.get_result(
            async_result=async_result,
            logger=StdOutputLogger(),
            timeout=TASKS_TIMEOUT,
        )
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
    assert view_info.name in views

    view_intended_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )
    assert view_intended_schema == view_info.schema_

    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view_info.name,
    )
    view_data_json = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )
    view_data = TableData.parse_raw(view_data_json)
    assert len(view_data.columns) == len(view_intended_schema.columns)
    assert view_data.name == view_info.name


@pytest.mark.slow
def test_view_with_filters(
    request_id,
    context_id,
    load_data_localnode1,
    localnode1_node_service,
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
    table_info = TableInfo.parse_raw(
        localnode1_celery_app.get_result(
            async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
        )
    )

    values = [[1, 0.1, "test1"], [2, 0.2, None], [3, 0.3, "test3"]]
    # task_signature = get_celery_task_signature("insert_data_to_table")
    # async_result = localnode1_celery_app.queue_task(
    #     task_signature=task_signature,
    #     logger=StdOutputLogger(),
    #     request_id=request_id,
    #     table_name=table_info.name,
    #     values=values,
    # )
    # localnode1_celery_app.get_result(
    #     async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    # )
    insert_data_to_localnode(table_info.name, values, MONETDB_LOCALNODE1_PORT)

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
        table_name=table_info.name,
        columns=columns,
        filters=filters,
    )
    view_info = TableInfo.parse_raw(
        localnode1_celery_app.get_result(
            async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
        )
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
    assert view_info.name in views
    view_intended_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )
    assert view_intended_schema == view_info.schema_

    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view_info.name,
    )
    view_data_json = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )
    view_data = TableData.parse_raw(view_data_json)
    assert len(view_data.columns) == 2
    assert len(view_data.columns) == len(view_intended_schema.columns)
    assert view_data.name == view_info.name


@pytest.mark.slow
def test_data_model_view_without_filters(
    request_id,
    context_id,
    load_data_localnode1,
    localnode1_node_service,
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
    view_info, *_ = [
        TableInfo.parse_raw(table)
        for table in localnode1_celery_app.get_result(
            async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
        )
    ]
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
    assert view_info.name in views

    schema = TableSchema(
        columns=[
            ColumnInfo(name="row_id", dtype=DType.INT),
            ColumnInfo(name="dataset", dtype=DType.STR),
            ColumnInfo(name="age_value", dtype=DType.INT),
            ColumnInfo(name="gcs_motor_response_scale", dtype=DType.STR),
            ColumnInfo(name="pupil_reactivity_right_eye_result", dtype=DType.STR),
        ]
    )
    assert schema == view_info.schema_

    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view_info.name,
    )
    view_data_json = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )
    view_data = TableData.parse_raw(view_data_json)
    assert len(view_data.columns) == len(schema.columns)
    assert view_data.name == view_info.name


@pytest.mark.slow
def test_data_model_view_with_filters(
    request_id,
    context_id,
    load_data_localnode1,
    localnode1_node_service,
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
    view_info, *_ = [
        TableInfo.parse_raw(table)
        for table in localnode1_celery_app.get_result(
            async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
        )
    ]

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
    assert view_info.name in views

    schema = TableSchema(
        columns=[
            ColumnInfo(name="row_id", dtype=DType.INT),
            ColumnInfo(name="dataset", dtype=DType.STR),
            ColumnInfo(name="age_value", dtype=DType.INT),
            ColumnInfo(name="gcs_motor_response_scale", dtype=DType.STR),
            ColumnInfo(name="pupil_reactivity_right_eye_result", dtype=DType.STR),
        ]
    )
    assert schema == view_info.schema_

    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view_info.name,
    )
    view_data_json = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )
    view_data = TableData.parse_raw(view_data_json)
    assert len(view_data.columns) == len(schema.columns)
    assert view_data.name == view_info.name


@pytest.mark.slow
def test_data_model_view_dataset_constraint(
    request_id,
    context_id,
    load_data_localnode1,
    localnode1_node_service,
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
    view_info, *_ = [
        TableInfo.parse_raw(table)
        for table in localnode1_celery_app.get_result(
            async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
        )
    ]

    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view_info.name,
    )
    view_data_json = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )

    _, dataset_column = TableData.parse_raw(view_data_json).columns
    assert set(dataset_column.data) == {"dummy_tbi1"}


@pytest.mark.slow
def test_data_model_view_null_constraints(
    request_id,
    context_id,
    load_data_localnode1,
    localnode1_node_service,
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
    view_info_without_nulls, *_ = [
        TableInfo.parse_raw(table)
        for table in localnode1_celery_app.get_result(
            async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
        )
    ]

    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view_info_without_nulls.name,
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
    view_info_with_nulls, *_ = [
        TableInfo.parse_raw(table)
        for table in localnode1_celery_app.get_result(
            async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
        )
    ]

    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view_info_with_nulls.name,
    )
    view_data_json = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )
    _, gose_score_column = TableData.parse_raw(view_data_json).columns

    assert None in gose_score_column.data


@pytest.mark.slow
def test_insuficient_data_error_raised_when_data_model_view_generated(
    request_id,
    context_id,
    load_data_localnode1,
    localnode1_node_service,
    localnode1_celery_app,
    use_localnode1_database,
    zero_rows_data_model_view_generating_params,
    five_rows_data_model_view_generating_params,
):
    # check InsufficientDataError raised when empty data model view generated
    with pytest.raises(InsufficientDataError):
        task_signature = get_celery_task_signature("create_data_model_views")
        async_result = localnode1_celery_app.queue_task(
            task_signature=task_signature,
            logger=StdOutputLogger(),
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            data_model=zero_rows_data_model_view_generating_params["data_model"],
            datasets=[],
            columns_per_view=[zero_rows_data_model_view_generating_params["columns"]],
            filters=zero_rows_data_model_view_generating_params["filters"],
        )
        localnode1_celery_app.get_result(
            async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
        )

    # check InsufficientDataError raised when data model view with less than
    # minimum_row_count (defined in testing_env_configs/test_localnode1.toml)
    with pytest.raises(InsufficientDataError):
        task_signature = get_celery_task_signature("create_data_model_views")
        async_result = localnode1_celery_app.queue_task(
            task_signature=task_signature,
            logger=StdOutputLogger(),
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            data_model=five_rows_data_model_view_generating_params["data_model"],
            datasets=[],
            columns_per_view=[five_rows_data_model_view_generating_params["columns"]],
            filters=five_rows_data_model_view_generating_params["filters"],
        )
        localnode1_celery_app.get_result(
            async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
        )


@pytest.mark.slow
def test_data_model_view_with_data_model_unavailable_exception(
    request_id,
    context_id,
    load_data_localnode1,
    localnode1_node_service,
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


@pytest.mark.slow
def test_data_model_view_with_dataset_unavailable_exception(
    request_id,
    context_id,
    load_data_localnode1,
    localnode1_node_service,
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


@pytest.mark.slow
def test_multiple_data_model_views(
    request_id,
    context_id,
    load_data_localnode1,
    localnode1_node_service,
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
    view1_info, view2_info = [
        TableInfo.parse_raw(result)
        for result in localnode1_celery_app.get_result(
            async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
        )
    ]

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
    assert view1_info.name in views
    assert view2_info.name in views
    schema1 = TableSchema(
        columns=[
            ColumnInfo(name="row_id", dtype=DType.INT),
            ColumnInfo(name="age_value", dtype=DType.INT),
            ColumnInfo(name="gcs_motor_response_scale", dtype=DType.STR),
        ]
    )
    schema2 = TableSchema(
        columns=[
            ColumnInfo(name="row_id", dtype=DType.INT),
            ColumnInfo(name="dataset", dtype=DType.STR),
            ColumnInfo(name="pupil_reactivity_right_eye_result", dtype=DType.STR),
        ]
    )
    assert schema1 == view1_info.schema_
    assert schema2 == view2_info.schema_


# NOTES:
# 1. this test (as well as other tests in the module) does not need the celery layer.
# Calling the task functions through queuing the task adds a lot of complexity that is
# unnecessary. The task functions should be tested by calling them as normal function
# from the relevant modules ex. tasks/views.py module and
# 2.instead of searching in the primary data tables for data that would fit the test,
# there should be a mechanism to add specifically crafted data for each test case in the
# db of the node
@pytest.mark.slow
def test_multiple_data_model_views_null_constraints(
    request_id,
    context_id,
    load_data_localnode1,
    localnode1_node_service,
    localnode1_celery_app,
    use_localnode1_database,
):
    # datasets:"edsd2" of the data model dementia:0.1 has 53 rows. Nevertheless
    # column 'neurodegenerativescategories' contains 20 NULL values, whereas 'opticchiasm',
    # for the same respective rows contains numerical values
    # The test checks that calling task "create_data_model_view" with "dropna" flag set to
    # true will create 2 "data model views" without the NULL values but keeping the two
    # columns rows aligned

    data_model = "dementia:0.1"
    datasets = ["edsd2"]
    neurodegenerativescategories_column_name = "neurodegenerativescategories"
    opticchiasm_column_name = "opticchiasm"
    columns_per_view = [
        [
            neurodegenerativescategories_column_name,
        ],
        [
            opticchiasm_column_name,
        ],
    ]

    # Create data model passing only column 'neurodegenerativescategories' with dropna=False
    # This data model view will contain null values
    task_signature = get_celery_task_signature("create_data_model_views")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        data_model=data_model,
        datasets=datasets,
        columns_per_view=[[neurodegenerativescategories_column_name]],
        filters=None,
        dropna=False,
        check_min_rows=False,
    )
    result = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )

    view_neurodegenerativescategories_nulls_included = TableInfo.parse_raw(result[0])

    # Get count of how many null values
    task_signature = get_celery_task_signature("get_table_data")
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view_neurodegenerativescategories_nulls_included.name,
    )
    result = localnode1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )
    _, view_neurodegenerativescategories_nulls_included_column = TableData.parse_raw(
        result
    ).columns
    view_neurodegenerativescategories_nulls_included_num_of_nulls = (
        view_neurodegenerativescategories_nulls_included_column.data.count(None)
    )
    view_neurodegenerativescategories_nulls_included_num_of_rows = len(
        view_neurodegenerativescategories_nulls_included_column.data
    )

    # Create data model with of both columns and dropna=True per view will drop the
    # rows with null values
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
        dropna=True,
        check_min_rows=False,
    )
    view_neurodegenerativescategories, view_opticchiasm = [
        TableInfo.parse_raw(result)
        for result in localnode1_celery_app.get_result(
            async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
        )
    ]

    # Get data of the created data model views
    task_signature = get_celery_task_signature("get_table_data")
    # neurodegenerativescategories data model view
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view_neurodegenerativescategories.name,
    )
    result = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )

    _, neurodegenerativescategories_column = TableData.parse_raw(result).columns
    view_neurodegenerativescategories_num_of_rows = len(
        neurodegenerativescategories_column.data
    )

    # opticchiasm data model view
    async_result = localnode1_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=view_opticchiasm.name,
    )
    result = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )
    _, opticchiasm_column = TableData.parse_raw(result).columns
    view_opticchiasm_num_of_rows = len(opticchiasm_column.data)

    # check that the 2 data model views generated are of the same row count
    assert view_neurodegenerativescategories_num_of_rows == view_opticchiasm_num_of_rows
    # check that the number of null rows excluded are as expected
    assert (
        view_neurodegenerativescategories_num_of_rows
        == view_neurodegenerativescategories_nulls_included_num_of_rows
        - view_neurodegenerativescategories_nulls_included_num_of_nulls
    )


@pytest.mark.slow
def test_bad_filters_exception(controller_service_with_localnode1):
    algorithm_name = "standard_deviation"
    request_params = {
        "inputdata": {
            "data_model": "dementia:0.1",
            "datasets": ["edsd0"],
            "x": [
                "lefthippocampus",
            ],
            "filters": {"whateveeeeeer": "!!!"},
        },
    }

    algorithm_url = ALGORITHMS_URL + "/" + algorithm_name
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        algorithm_url,
        data=json.dumps(request_params),
        headers=headers,
    )

    assert "Invalid filters format." in response.text
    assert response.status_code == 400
