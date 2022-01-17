import json
import uuid

import pytest
from typing import Tuple

from mipengine import DType
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import NodeSMPCValueDTO
from mipengine.node_tasks_DTOs import NodeSMPCDTO
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import NodeTableDTO
from mipengine.node_tasks_DTOs import UDFKeyArguments
from mipengine.node_tasks_DTOs import UDFPosArguments
from mipengine.node_tasks_DTOs import UDFResults
from mipengine.udfgen import make_unique_func_name
from tests.algorithms.orphan_udfs import smpc_global_step
from tests.algorithms.orphan_udfs import smpc_local_step
from tests.standalone_tests.conftest import LOCALNODE_1_CONFIG_FILE
from tests.standalone_tests.conftest import LOCALNODE_SMPC_CONFIG_FILE
from tests.standalone_tests.nodes_communication_helper import get_celery_app
from tests.standalone_tests.nodes_communication_helper import get_celery_task_signature
from tests.standalone_tests.nodes_communication_helper import get_node_config_by_id
from tests.standalone_tests.test_udfs import create_table_with_one_column_and_ten_rows


context_id = "test_smpc_udfs_" + str(uuid.uuid4().hex)[:10]
command_id = "command123"


@pytest.fixture(scope="session")
def localnode_1_celery_app():
    localnode1_config = get_node_config_by_id(LOCALNODE_1_CONFIG_FILE)
    yield get_celery_app(localnode1_config)


@pytest.fixture(scope="session")
def smpc_localnode_celery_app():
    localnode1_config = get_node_config_by_id(LOCALNODE_SMPC_CONFIG_FILE)
    yield get_celery_app(localnode1_config)


# TODO More dynamic so it can receive any secure_transfer values
def create_table_with_secure_transfer_results(celery_app) -> Tuple[str, int]:
    create_table_task = get_celery_task_signature(celery_app, "create_table")
    insert_data_to_table_task = get_celery_task_signature(
        celery_app, "insert_data_to_table"
    )

    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="secure_transfer", dtype=DType.JSON),
        ]
    )
    table_name = create_table_task.delay(
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    ).get()

    secure_transfer_1_value = 100
    secure_transfer_2_value = 11

    secure_transfer_1 = {
        "sum": {"data": secure_transfer_1_value, "type": "int", "operation": "addition"}
    }
    secure_transfer_2 = {
        "sum": {"data": secure_transfer_2_value, "type": "int", "operation": "addition"}
    }
    values = [[json.dumps(secure_transfer_1)], [json.dumps(secure_transfer_2)]]
    insert_data_to_table_task.delay(table_name=table_name, values=values).get()

    return table_name, secure_transfer_1_value + secure_transfer_2_value


def create_table_with_multiple_secure_transfer_templates(
    celery_app, similar: bool
) -> str:
    create_table_task = get_celery_task_signature(celery_app, "create_table")
    insert_data_to_table_task = get_celery_task_signature(
        celery_app, "insert_data_to_table"
    )
    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="secure_transfer", dtype=DType.JSON),
        ]
    )
    table_name = create_table_task.delay(
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    ).get()
    secure_transfer_template = {
        "sum": {"data": [0, 1, 2, 3], "type": "int", "operation": "addition"}
    }
    differenet_secure_transfer_template = {
        "sum": {"data": 0, "type": "int", "operation": "addition"}
    }

    if similar:
        values = [
            [json.dumps(secure_transfer_template)],
            [json.dumps(secure_transfer_template)],
        ]
    else:
        values = [
            [json.dumps(secure_transfer_template)],
            [json.dumps(differenet_secure_transfer_template)],
        ]

    insert_data_to_table_task.delay(table_name=table_name, values=values).get()

    return table_name


def validate_dict_table_data_match_expected(
    get_table_data_task, table_name, expected_values
):
    assert table_name is not None
    table_data_str = get_table_data_task.delay(table_name=table_name).get()
    table_data: TableData = TableData.parse_raw(table_data_str)
    result_str, *_ = table_data.columns[1].data
    result = json.loads(result_str)
    assert result == expected_values


def test_secure_transfer_output_with_smpc_off(
    localnode_1_node_service, use_localnode_1_database, localnode_1_celery_app
):
    run_udf_task = get_celery_task_signature(localnode_1_celery_app, "run_udf")
    get_table_data_task = get_celery_task_signature(
        localnode_1_celery_app, "get_table_data"
    )

    input_table_name, input_table_name_sum = create_table_with_one_column_and_ten_rows(
        localnode_1_celery_app
    )

    pos_args_str = UDFPosArguments(args=[NodeTableDTO(value=input_table_name)]).json()

    udf_results_str = run_udf_task.delay(
        command_id="1",
        context_id=context_id,
        func_name=make_unique_func_name(smpc_local_step),
        positional_args_json=pos_args_str,
        keyword_args_json=UDFKeyArguments(args={}).json(),
    ).get()

    results = UDFResults.parse_raw(udf_results_str).results
    assert len(results) == 1

    secure_transfer_result = results[0]
    assert isinstance(secure_transfer_result, NodeTableDTO)

    expected_result = {
        "sum": {"data": input_table_name_sum, "type": "int", "operation": "addition"}
    }
    validate_dict_table_data_match_expected(
        get_table_data_task,
        secure_transfer_result.value,
        expected_result,
    )


def test_secure_transfer_input_with_smpc_off(
    localnode_1_node_service, use_localnode_1_database, localnode_1_celery_app
):
    run_udf_task = get_celery_task_signature(localnode_1_celery_app, "run_udf")
    get_table_data_task = get_celery_task_signature(
        localnode_1_celery_app, "get_table_data"
    )

    (
        secure_transfer_results_tablename,
        secure_transfer_results_values_sum,
    ) = create_table_with_secure_transfer_results(localnode_1_celery_app)

    pos_args_str = UDFPosArguments(
        args=[NodeTableDTO(value=secure_transfer_results_tablename)]
    ).json()

    udf_results_str = run_udf_task.delay(
        command_id="1",
        context_id=context_id,
        func_name=make_unique_func_name(smpc_global_step),
        positional_args_json=pos_args_str,
        keyword_args_json=UDFKeyArguments(args={}).json(),
    ).get()

    results = UDFResults.parse_raw(udf_results_str).results
    assert len(results) == 1

    transfer_result = results[0]
    assert isinstance(transfer_result, NodeTableDTO)

    expected_result = {"total_sum": secure_transfer_results_values_sum}
    validate_dict_table_data_match_expected(
        get_table_data_task,
        transfer_result.value,
        expected_result,
    )


def test_secure_transfer_flow_with_smpc_on(
    smpc_localnode_node_service, use_localnode_1_database, smpc_localnode_celery_app
):
    run_udf_task = get_celery_task_signature(smpc_localnode_celery_app, "run_udf")
    get_table_data_task = get_celery_task_signature(
        smpc_localnode_celery_app, "get_table_data"
    )

    # ----------------------- SECURE TRANSFER OUTPUT ----------------------

    input_table_name, input_table_name_sum = create_table_with_one_column_and_ten_rows(
        smpc_localnode_celery_app
    )

    pos_args_str = UDFPosArguments(args=[NodeTableDTO(value=input_table_name)]).json()

    udf_results_str = run_udf_task.delay(
        command_id="1",
        context_id=context_id,
        func_name=make_unique_func_name(smpc_local_step),
        positional_args_json=pos_args_str,
        keyword_args_json=UDFKeyArguments(args={}).json(),
        use_smpc=True,
    ).get()

    local_step_results = UDFResults.parse_raw(udf_results_str).results
    assert len(local_step_results) == 1

    smpc_result = local_step_results[0]
    assert isinstance(smpc_result, NodeSMPCDTO)

    assert smpc_result.value.template is not None
    expected_template = {"sum": {"data": 0, "type": "int", "operation": "addition"}}
    validate_dict_table_data_match_expected(
        get_table_data_task,
        smpc_result.value.template.value,
        expected_template,
    )

    assert smpc_result.value.add_op_values is not None
    expected_add_op_values = [input_table_name_sum]
    validate_dict_table_data_match_expected(
        get_table_data_task,
        smpc_result.value.add_op_values.value,
        expected_add_op_values,
    )

    # ----------------------- SECURE TRANSFER INPUT----------------------

    # Providing as input the smpc_result created from the previous udf (local step)
    smpc_arg = NodeSMPCDTO(
        value=NodeSMPCValueDTO(
            template=NodeTableDTO(value=smpc_result.value.template.value),
            add_op_values=NodeTableDTO(value=smpc_result.value.add_op_values.value),
        )
    )

    pos_args_str = UDFPosArguments(args=[smpc_arg]).json()

    udf_results_str = run_udf_task.delay(
        command_id="2",
        context_id=context_id,
        func_name=make_unique_func_name(smpc_global_step),
        positional_args_json=pos_args_str,
        keyword_args_json=UDFKeyArguments(args={}).json(),
        use_smpc=True,
    ).get()

    global_step_results = UDFResults.parse_raw(udf_results_str).results
    assert len(global_step_results) == 1

    global_step_result = global_step_results[0]
    assert isinstance(global_step_result, NodeTableDTO)

    expected_result = {"total_sum": input_table_name_sum}
    validate_dict_table_data_match_expected(
        get_table_data_task,
        global_step_result.value,
        expected_result,
    )


def test_validate_smpc_templates_match(
    smpc_localnode_node_service, use_localnode_1_database, smpc_localnode_celery_app
):
    validate_smpc_templates_match_task = get_celery_task_signature(
        smpc_localnode_celery_app, "validate_smpc_templates_match"
    )

    table_name = create_table_with_multiple_secure_transfer_templates(
        smpc_localnode_celery_app, True
    )

    try:
        validate_smpc_templates_match_task.delay(
            context_id=context_id, table_name=table_name
        ).get()
    except Exception as exc:
        pytest.fail(f"No exception should be raised. Exception: {exc}")


def test_validate_smpc_templates_dont_match(
    smpc_localnode_node_service, use_localnode_1_database, smpc_localnode_celery_app
):
    validate_smpc_templates_match_task = get_celery_task_signature(
        smpc_localnode_celery_app, "validate_smpc_templates_match"
    )

    table_name = create_table_with_multiple_secure_transfer_templates(
        smpc_localnode_celery_app, False
    )

    with pytest.raises(ValueError) as exc:
        validate_smpc_templates_match_task.delay(
            context_id=context_id, table_name=table_name
        ).get()
    assert "SMPC templates dont match." in str(exc)
