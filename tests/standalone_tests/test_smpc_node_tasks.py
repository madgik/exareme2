import json
import uuid
from time import sleep
from typing import Any
from typing import Tuple

import pytest
import requests

from exareme2 import DType
from exareme2.algorithms.in_database.udfgen import make_unique_func_name
from exareme2.node_communication import ColumnInfo
from exareme2.node_communication import NodeSMPCDTO
from exareme2.node_communication import NodeTableDTO
from exareme2.node_communication import NodeUDFKeyArguments
from exareme2.node_communication import NodeUDFPosArguments
from exareme2.node_communication import NodeUDFResults
from exareme2.node_communication import SMPCTablesInfo
from exareme2.node_communication import TableInfo
from exareme2.node_communication import TableSchema
from exareme2.node_communication import TableType
from exareme2.smpc_cluster_communication import ADD_DATASET_ENDPOINT
from exareme2.smpc_cluster_communication import TRIGGER_COMPUTATION_ENDPOINT
from exareme2.smpc_cluster_communication import SMPCRequestData
from exareme2.smpc_cluster_communication import SMPCRequestType
from exareme2.smpc_cluster_communication import SMPCResponse
from exareme2.smpc_cluster_communication import SMPCResponseStatus
from exareme2.smpc_cluster_communication import get_smpc_result
from tests.algorithms.orphan_udfs import smpc_global_step
from tests.algorithms.orphan_udfs import smpc_local_step
from tests.standalone_tests.conftest import LOCALNODE1_SMPC_CONFIG_FILE
from tests.standalone_tests.conftest import LOCALNODE2_SMPC_CONFIG_FILE
from tests.standalone_tests.conftest import SMPC_COORDINATOR_ADDRESS
from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.conftest import create_table_in_db
from tests.standalone_tests.conftest import get_node_config_by_id
from tests.standalone_tests.conftest import get_table_data_from_db
from tests.standalone_tests.conftest import insert_data_to_db
from tests.standalone_tests.nodes_communication_helper import get_celery_task_signature

request_id = "testsmpcudfs" + str(uuid.uuid4().hex)[:10] + "request"
context_id = "testsmpcudfs" + str(uuid.uuid4().hex)[:10]
command_id = "command123"
smpc_job_id = "testKey123"
SMPC_GET_DATASET_ENDPOINT = "/api/update-dataset/"


def create_table_with_one_column_and_ten_rows(db_cursor) -> Tuple[TableInfo, int]:
    table_name = f"table_one_column_ten_rows_{context_id}"
    table_info = TableInfo(
        name=table_name,
        schema_=TableSchema(
            columns=[
                ColumnInfo(name="col1", dtype=DType.INT),
            ]
        ),
        type_=TableType.NORMAL,
    )
    create_table_in_db(db_cursor, table_info.name, table_info.schema_)

    values = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]

    insert_data_to_db(table_info.name, values, db_cursor)

    return table_info, 55


def create_secure_transfer_table(db_cursor) -> TableInfo:
    table_name = f"table_secure_transfer_{context_id}"
    table_info = TableInfo(
        name=table_name,
        schema_=TableSchema(
            columns=[
                ColumnInfo(name="secure_transfer", dtype=DType.JSON),
            ]
        ),
        type_=TableType.NORMAL,
    )
    create_table_in_db(db_cursor, table_info.name, table_info.schema_)

    return table_info


# TODO More dynamic so it can receive any secure_transfer values
def create_table_with_secure_transfer_results_with_smpc_off(
    db_cursor,
) -> Tuple[TableInfo, int]:
    table_info = create_secure_transfer_table(db_cursor)

    secure_transfer_1_value = 100
    secure_transfer_2_value = 11

    secure_transfer_1 = {
        "sum": {"data": secure_transfer_1_value, "operation": "sum", "type": "int"}
    }
    secure_transfer_2 = {
        "sum": {"data": secure_transfer_2_value, "operation": "sum", "type": "int"}
    }
    values = [
        [json.dumps(secure_transfer_1)],
        [json.dumps(secure_transfer_2)],
    ]

    insert_data_to_db(table_info.name, values, db_cursor)

    return table_info, secure_transfer_1_value + secure_transfer_2_value


def create_table_with_multiple_secure_transfer_templates(
    db_cursor, similar: bool
) -> TableInfo:
    table_info = create_secure_transfer_table(db_cursor)

    secure_transfer_template = {
        "sum": {"data": [0, 1, 2, 3], "operation": "sum", "type": "int"}
    }
    different_secure_transfer_template = {
        "sum": {"data": 0, "operation": "sum", "type": "int"}
    }

    if similar:
        values = [
            [json.dumps(secure_transfer_template)],
            [json.dumps(secure_transfer_template)],
        ]
    else:
        values = [
            [json.dumps(secure_transfer_template)],
            [json.dumps(different_secure_transfer_template)],
        ]

    insert_data_to_db(table_info.name, values, db_cursor)

    return table_info


def create_table_with_smpc_sum_op_values(db_cursor) -> Tuple[TableInfo, str]:
    table_info = create_secure_transfer_table(db_cursor)

    sum_op_values = [0, 1, 2, 3, 4, 5]
    values = [
        [json.dumps(sum_op_values)],
    ]

    insert_data_to_db(table_info.name, values, db_cursor)

    return table_info, json.dumps(sum_op_values)


def validate_table_data_match_expected(
    db_cursor,
    table_name: str,
    expected_values: Any,
):
    assert table_name is not None

    table_data = get_table_data_from_db(db_cursor, table_name)
    result = json.loads(table_data[0][0])
    assert result == expected_values


@pytest.mark.slow
def test_secure_transfer_output_with_smpc_off(
    localnode1_node_service,
    use_localnode1_database,
    localnode1_celery_app,
    localnode1_db_cursor,
):
    localnode1_celery_app = localnode1_celery_app._celery_app
    run_udf_task = get_celery_task_signature("run_udf")

    input_table_info, input_table_name_sum = create_table_with_one_column_and_ten_rows(
        localnode1_db_cursor
    )

    pos_args_str = NodeUDFPosArguments(
        args=[NodeTableDTO(value=input_table_info)]
    ).json()
    udf_results_str = (
        localnode1_celery_app.signature(run_udf_task)
        .delay(
            command_id="1",
            request_id=request_id,
            context_id=context_id,
            func_name=make_unique_func_name(smpc_local_step),
            positional_args_json=pos_args_str,
            keyword_args_json=NodeUDFKeyArguments(args={}).json(),
        )
        .get(timeout=TASKS_TIMEOUT)
    )

    results = NodeUDFResults.parse_raw(udf_results_str).results
    assert len(results) == 1

    secure_transfer_result = results[0]
    assert isinstance(secure_transfer_result, NodeTableDTO)

    expected_result = {
        "sum": {"data": input_table_name_sum, "operation": "sum", "type": "int"}
    }
    validate_table_data_match_expected(
        db_cursor=localnode1_db_cursor,
        table_name=secure_transfer_result.value.name,
        expected_values=expected_result,
    )


@pytest.mark.slow
def test_secure_transfer_input_with_smpc_off(
    localnode1_node_service,
    use_localnode1_database,
    localnode1_celery_app,
    localnode1_db_cursor,
):
    localnode1_celery_app = localnode1_celery_app._celery_app
    run_udf_task = get_celery_task_signature("run_udf")

    (
        secure_transfer_results_tableinfo,
        secure_transfer_results_values_sum,
    ) = create_table_with_secure_transfer_results_with_smpc_off(localnode1_db_cursor)

    pos_args_str = NodeUDFPosArguments(
        args=[NodeTableDTO(value=secure_transfer_results_tableinfo)]
    ).json()

    udf_results_str = (
        localnode1_celery_app.signature(run_udf_task)
        .delay(
            command_id="1",
            request_id=request_id,
            context_id=context_id,
            func_name=make_unique_func_name(smpc_global_step),
            positional_args_json=pos_args_str,
            keyword_args_json=NodeUDFKeyArguments(args={}).json(),
        )
        .get(timeout=TASKS_TIMEOUT)
    )

    results = NodeUDFResults.parse_raw(udf_results_str).results
    assert len(results) == 1

    transfer_result = results[0]
    assert isinstance(transfer_result, NodeTableDTO)

    expected_result = {"total_sum": secure_transfer_results_values_sum}
    validate_table_data_match_expected(
        db_cursor=localnode1_db_cursor,
        table_name=transfer_result.value.name,
        expected_values=expected_result,
    )


@pytest.mark.slow
@pytest.mark.very_slow
@pytest.mark.smpc
def test_validate_smpc_templates_match(
    smpc_localnode1_node_service,
    use_smpc_localnode1_database,
    smpc_localnode1_celery_app,
    localnode1_smpc_db_cursor,
):
    smpc_localnode1_celery_app = smpc_localnode1_celery_app._celery_app
    validate_smpc_templates_match_task = get_celery_task_signature(
        "validate_smpc_templates_match"
    )

    table_info = create_table_with_multiple_secure_transfer_templates(
        localnode1_smpc_db_cursor, True
    )

    try:
        smpc_localnode1_celery_app.signature(validate_smpc_templates_match_task).delay(
            request_id=request_id,
            table_name=table_info.name,
        ).get(timeout=TASKS_TIMEOUT)
    except Exception as exc:
        pytest.fail(f"No exception should be raised. Exception: {exc}")


@pytest.mark.slow
@pytest.mark.very_slow
@pytest.mark.smpc
def test_validate_smpc_templates_dont_match(
    smpc_localnode1_node_service,
    use_smpc_localnode1_database,
    smpc_localnode1_celery_app,
    localnode1_smpc_db_cursor,
):
    smpc_localnode1_celery_app = smpc_localnode1_celery_app._celery_app
    validate_smpc_templates_match_task = get_celery_task_signature(
        "validate_smpc_templates_match"
    )

    table_info = create_table_with_multiple_secure_transfer_templates(
        localnode1_smpc_db_cursor, False
    )

    with pytest.raises(ValueError) as exc:
        smpc_localnode1_celery_app.signature(validate_smpc_templates_match_task).delay(
            request_id=request_id,
            table_name=table_info.name,
        ).get(timeout=TASKS_TIMEOUT)
    assert "SMPC templates dont match." in str(exc)


@pytest.mark.slow
@pytest.mark.very_slow
@pytest.mark.smpc
def test_secure_transfer_run_udf_flow_with_smpc_on(
    smpc_localnode1_node_service,
    use_smpc_localnode1_database,
    smpc_localnode1_celery_app,
    localnode1_smpc_db_cursor,
):
    smpc_localnode1_celery_app = smpc_localnode1_celery_app._celery_app
    run_udf_task = get_celery_task_signature("run_udf")

    # ----------------------- SECURE TRANSFER OUTPUT ----------------------
    input_table_name, input_table_name_sum = create_table_with_one_column_and_ten_rows(
        localnode1_smpc_db_cursor
    )

    pos_args_str = NodeUDFPosArguments(
        args=[NodeTableDTO(value=input_table_name)]
    ).json()

    udf_results_str = (
        smpc_localnode1_celery_app.signature(run_udf_task)
        .delay(
            command_id="1",
            request_id=request_id,
            context_id=context_id,
            func_name=make_unique_func_name(smpc_local_step),
            positional_args_json=pos_args_str,
            keyword_args_json=NodeUDFKeyArguments(args={}).json(),
            use_smpc=True,
        )
        .get(timeout=TASKS_TIMEOUT)
    )

    local_step_results = NodeUDFResults.parse_raw(udf_results_str).results
    assert len(local_step_results) == 1

    smpc_result = local_step_results[0]
    assert isinstance(smpc_result, NodeSMPCDTO)

    assert smpc_result.value.template is not None
    expected_template = {"sum": {"data": 0, "operation": "sum", "type": "int"}}
    validate_table_data_match_expected(
        db_cursor=localnode1_smpc_db_cursor,
        table_name=smpc_result.value.template.name,
        expected_values=expected_template,
    )

    assert smpc_result.value.sum_op is not None
    expected_sum_op_values = [input_table_name_sum]
    validate_table_data_match_expected(
        db_cursor=localnode1_smpc_db_cursor,
        table_name=smpc_result.value.sum_op.name,
        expected_values=expected_sum_op_values,
    )

    # ----------------------- SECURE TRANSFER INPUT----------------------
    pos_args_str = NodeUDFPosArguments(args=[smpc_result]).json()

    udf_results_str = (
        smpc_localnode1_celery_app.signature(run_udf_task)
        .delay(
            command_id="2",
            request_id=request_id,
            context_id=context_id,
            func_name=make_unique_func_name(smpc_global_step),
            positional_args_json=pos_args_str,
            keyword_args_json=NodeUDFKeyArguments(args={}).json(),
            use_smpc=True,
        )
        .get(timeout=TASKS_TIMEOUT)
    )

    global_step_results = NodeUDFResults.parse_raw(udf_results_str).results
    assert len(global_step_results) == 1

    global_step_result = global_step_results[0]
    assert isinstance(global_step_result, NodeTableDTO)

    expected_result = {"total_sum": input_table_name_sum}
    validate_table_data_match_expected(
        db_cursor=localnode1_smpc_db_cursor,
        table_name=global_step_result.value.name,
        expected_values=expected_result,
    )


@pytest.mark.slow
@pytest.mark.very_slow
@pytest.mark.smpc
def test_load_data_to_smpc_client_from_globalnode_fails(
    smpc_globalnode_node_service,
    smpc_globalnode_celery_app,
):
    smpc_globalnode_celery_app = smpc_globalnode_celery_app._celery_app
    load_data_to_smpc_client_task = get_celery_task_signature(
        "load_data_to_smpc_client"
    )

    with pytest.raises(PermissionError) as exc:
        smpc_globalnode_celery_app.signature(load_data_to_smpc_client_task).delay(
            request_id=request_id,
            table_name="whatever",
            jobid="whatever",
        ).get(timeout=TASKS_TIMEOUT)
    assert "load_data_to_smpc_client is allowed only for a LOCALNODE." in str(exc)


@pytest.mark.slow
@pytest.mark.very_slow
@pytest.mark.smpc
@pytest.mark.smpc_cluster
def test_load_data_to_smpc_client(
    smpc_localnode1_node_service,
    use_smpc_localnode1_database,
    smpc_localnode1_celery_app,
    localnode1_smpc_db_cursor,
    smpc_cluster,
):
    smpc_localnode1_celery_app = smpc_localnode1_celery_app._celery_app
    table_info, sum_op_values_str = create_table_with_smpc_sum_op_values(
        localnode1_smpc_db_cursor
    )
    load_data_to_smpc_client_task = get_celery_task_signature(
        "load_data_to_smpc_client"
    )

    smpc_localnode1_celery_app.signature(load_data_to_smpc_client_task).delay(
        request_id=request_id,
        table_name=table_info.name,
        jobid=smpc_job_id,
    ).get(timeout=TASKS_TIMEOUT)

    node_config = get_node_config_by_id(LOCALNODE1_SMPC_CONFIG_FILE)
    request_url = (
        node_config.smpc.client_address + SMPC_GET_DATASET_ENDPOINT + smpc_job_id
    )
    request_headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.get(
        request_url,
        headers=request_headers,
    )

    assert response.status_code == 200

    # This debugging call returns the values as strings
    str_result = json.loads(response.text)
    result = []
    for i in range(0, len(str_result), 2):
        result.append(int(str_result[i]))
        assert (
            str_result[i + 1] == 0
        )  # The even elements are the decimal parts, that should be zero
    assert json.dumps(result) == sum_op_values_str


@pytest.mark.slow
@pytest.mark.very_slow
@pytest.mark.smpc
def test_get_smpc_result_from_localnode_fails(
    smpc_localnode1_node_service,
    smpc_localnode1_celery_app,
):
    smpc_localnode1_celery_app = smpc_localnode1_celery_app._celery_app
    get_smpc_result_task = get_celery_task_signature("get_smpc_result")

    with pytest.raises(PermissionError) as exc:
        smpc_localnode1_celery_app.signature(get_smpc_result_task).delay(
            request_id="whatever",
            context_id="whatever",
            command_id="whatever",
            jobid="whatever",
        ).get(timeout=TASKS_TIMEOUT)
    assert "get_smpc_result is allowed only for a GLOBALNODE." in str(exc)


@pytest.mark.slow
@pytest.mark.very_slow
@pytest.mark.smpc
@pytest.mark.smpc_cluster
def test_get_smpc_result(
    smpc_globalnode_node_service,
    use_smpc_globalnode_database,
    smpc_globalnode_celery_app,
    globalnode_smpc_db_cursor,
    smpc_cluster,
):
    smpc_globalnode_celery_app = smpc_globalnode_celery_app._celery_app
    get_smpc_result_task = get_celery_task_signature("get_smpc_result")

    # --------------- LOAD Dataset to SMPC --------------------
    node_config = get_node_config_by_id(LOCALNODE1_SMPC_CONFIG_FILE)
    request_url = node_config.smpc.client_address + ADD_DATASET_ENDPOINT + smpc_job_id
    request_headers = {"Content-type": "application/json", "Accept": "text/plain"}
    smpc_computation_data = [100]
    response = requests.post(
        request_url,
        data=json.dumps({"type": "int", "data": smpc_computation_data}),
        headers=request_headers,
    )
    assert response.status_code == 200

    # --------------- Trigger Computation ------------------------
    request_url = SMPC_COORDINATOR_ADDRESS + TRIGGER_COMPUTATION_ENDPOINT + smpc_job_id
    request_headers = {"Content-type": "application/json", "Accept": "text/plain"}
    request_body = json.dumps(
        {"computationType": "sum", "clients": [node_config.smpc.client_id]}
    )
    response = requests.post(
        request_url,
        data=request_body,
        headers=request_headers,
    )
    assert response.status_code == 200

    # --------------- Wait for SMPC result to be ready ------------------------
    for _ in range(1, 100):
        response = get_smpc_result(
            coordinator_address=SMPC_COORDINATOR_ADDRESS,
            jobid=smpc_job_id,
        )
        smpc_response = SMPCResponse.parse_raw(response)

        if smpc_response.status == SMPCResponseStatus.FAILED:
            raise ValueError(
                f"The SMPC returned a {SMPCResponseStatus.FAILED} status. Body: {response}"
            )
        elif smpc_response.status == SMPCResponseStatus.COMPLETED:
            break
        sleep(1)
    else:
        raise TimeoutError("SMPC did not finish in 100 tries.")

    # --------------- GET SMPC RESULT IN GLOBALNODE ------------------------
    result_tableinfo = TableInfo.parse_raw(
        smpc_globalnode_celery_app.signature(get_smpc_result_task)
        .delay(
            request_id=request_id,
            context_id=context_id,
            command_id=command_id,
            jobid=smpc_job_id,
        )
        .get(timeout=TASKS_TIMEOUT)
    )

    validate_table_data_match_expected(
        db_cursor=globalnode_smpc_db_cursor,
        table_name=result_tableinfo.name,
        expected_values=smpc_computation_data,
    )


@pytest.mark.slow
@pytest.mark.very_slow
@pytest.mark.smpc
@pytest.mark.smpc_cluster
def test_orchestrate_SMPC_between_two_localnodes_and_the_globalnode(
    smpc_globalnode_node_service,
    smpc_localnode1_node_service,
    smpc_localnode2_node_service,
    use_smpc_globalnode_database,
    use_smpc_localnode1_database,
    use_smpc_localnode2_database,
    smpc_globalnode_celery_app,
    smpc_localnode1_celery_app,
    smpc_localnode2_celery_app,
    localnode1_smpc_db_cursor,
    localnode2_smpc_db_cursor,
    globalnode_smpc_db_cursor,
    smpc_cluster,
):
    smpc_globalnode_celery_app = smpc_globalnode_celery_app._celery_app
    smpc_localnode1_celery_app = smpc_localnode1_celery_app._celery_app
    smpc_localnode2_celery_app = smpc_localnode2_celery_app._celery_app

    run_udf_task_globalnode = smpc_globalnode_celery_app.signature(
        get_celery_task_signature("run_udf")
    )
    run_udf_task_localnode1 = smpc_localnode1_celery_app.signature(
        get_celery_task_signature("run_udf")
    )
    run_udf_task_localnode2 = smpc_localnode2_celery_app.signature(
        get_celery_task_signature("run_udf")
    )
    create_remote_table_task_globalnode = smpc_globalnode_celery_app.signature(
        get_celery_task_signature("create_remote_table")
    )
    create_merge_table_task_globalnode = smpc_globalnode_celery_app.signature(
        get_celery_task_signature("create_merge_table")
    )
    validate_smpc_templates_match_task_globalnode = (
        smpc_globalnode_celery_app.signature(
            get_celery_task_signature("validate_smpc_templates_match")
        )
    )
    load_data_to_smpc_client_task_localnode1 = smpc_localnode1_celery_app.signature(
        get_celery_task_signature("load_data_to_smpc_client")
    )
    load_data_to_smpc_client_task_localnode2 = smpc_localnode2_celery_app.signature(
        get_celery_task_signature("load_data_to_smpc_client")
    )
    get_smpc_result_task_globalnode = smpc_globalnode_celery_app.signature(
        get_celery_task_signature("get_smpc_result")
    )

    # ---------------- CREATE LOCAL TABLES WITH INITIAL DATA ----------------------
    (
        input_table_1_name,
        input_table_1_name_sum,
    ) = create_table_with_one_column_and_ten_rows(localnode1_smpc_db_cursor)
    (
        input_table_2_name,
        input_table_2_name_sum,
    ) = create_table_with_one_column_and_ten_rows(localnode2_smpc_db_cursor)

    # ---------------- RUN LOCAL UDFS WITH SECURE TRANSFER OUTPUT ----------------------
    pos_args_str_localnode1 = NodeUDFPosArguments(
        args=[NodeTableDTO(value=input_table_1_name)]
    ).json()
    pos_args_str_localnode2 = NodeUDFPosArguments(
        args=[NodeTableDTO(value=input_table_2_name)]
    ).json()

    udf_results_str_localnode1 = run_udf_task_localnode1.delay(
        command_id="1",
        request_id=request_id,
        context_id=context_id,
        func_name=make_unique_func_name(smpc_local_step),
        positional_args_json=pos_args_str_localnode1,
        keyword_args_json=NodeUDFKeyArguments(args={}).json(),
        use_smpc=True,
    ).get()

    udf_results_str_localnode2 = run_udf_task_localnode2.delay(
        command_id="2",
        request_id=request_id,
        context_id=context_id,
        func_name=make_unique_func_name(smpc_local_step),
        positional_args_json=pos_args_str_localnode2,
        keyword_args_json=NodeUDFKeyArguments(args={}).json(),
        use_smpc=True,
    ).get()

    local_1_smpc_result = NodeUDFResults.parse_raw(udf_results_str_localnode1).results[
        0
    ]
    assert isinstance(local_1_smpc_result, NodeSMPCDTO)
    local_2_smpc_result = NodeUDFResults.parse_raw(udf_results_str_localnode2).results[
        0
    ]
    assert isinstance(local_2_smpc_result, NodeSMPCDTO)

    # ---------- CREATE REMOTE/MERGE TABLE ON GLOBALNODE WITH SMPC TEMPLATE ---------
    localnode1_config = get_node_config_by_id(LOCALNODE1_SMPC_CONFIG_FILE)
    localnode2_config = get_node_config_by_id(LOCALNODE2_SMPC_CONFIG_FILE)

    localnode_1_monetdb_sock_address = (
        f"{str(localnode1_config.monetdb.ip)}:{localnode1_config.monetdb.port}"
    )
    localnode_2_monetdb_sock_address = (
        f"{str(localnode2_config.monetdb.ip)}:{localnode2_config.monetdb.port}"
    )
    create_remote_table_task_globalnode.delay(
        request_id=request_id,
        table_name=local_1_smpc_result.value.template.name,
        table_schema_json=local_1_smpc_result.value.template.schema_.json(),
        monetdb_socket_address=localnode_1_monetdb_sock_address,
    ).get()
    create_remote_table_task_globalnode.delay(
        request_id=request_id,
        table_name=local_2_smpc_result.value.template.name,
        table_schema_json=local_2_smpc_result.value.template.schema_.json(),
        monetdb_socket_address=localnode_2_monetdb_sock_address,
    ).get()
    globalnode_template_tableinfo = TableInfo.parse_raw(
        create_merge_table_task_globalnode.delay(
            request_id=request_id,
            context_id=context_id,
            command_id="3",
            table_infos_json=[
                local_1_smpc_result.value.template.json(),
                local_2_smpc_result.value.template.json(),
            ],
        ).get()
    )

    validate_smpc_templates_match_task_globalnode.delay(
        request_id=request_id,
        table_name=globalnode_template_tableinfo.name,
    ).get()

    # --------- LOAD LOCALNODE ADD OP DATA TO SMPC CLIENTS -----------------
    smpc_client_1 = load_data_to_smpc_client_task_localnode1.delay(
        request_id=request_id,
        table_name=local_1_smpc_result.value.sum_op.name,
        jobid=smpc_job_id,
    ).get()
    smpc_client_2 = load_data_to_smpc_client_task_localnode2.delay(
        request_id=request_id,
        table_name=local_2_smpc_result.value.sum_op.name,
        jobid=smpc_job_id,
    ).get()

    # --------- Trigger SMPC in the coordinator -----------------
    trigger_smpc_computation_url = (
        SMPC_COORDINATOR_ADDRESS + "/api/secure-aggregation/job-id/" + smpc_job_id
    )
    trigger_smpc_request_data = SMPCRequestData(
        computationType=SMPCRequestType.SUM.value,
        clients=[smpc_client_1, smpc_client_2],
    )
    trigger_smpc_request_headers = {
        "Content-type": "application/json",
        "Accept": "text/plain",
    }
    response = requests.post(
        url=trigger_smpc_computation_url,
        data=trigger_smpc_request_data.json(),
        headers=trigger_smpc_request_headers,
    )
    assert response.status_code == 200

    # --------------- Wait for SMPC result to be ready ------------------------
    for _ in range(1, 100):
        response = get_smpc_result(
            coordinator_address=SMPC_COORDINATOR_ADDRESS,
            jobid=smpc_job_id,
        )
        smpc_response = SMPCResponse.parse_raw(response)

        if smpc_response.status == SMPCResponseStatus.FAILED:
            raise ValueError(
                f"The SMPC returned a {SMPCResponseStatus.FAILED} status. Body: {response}"
            )
        elif smpc_response.status == SMPCResponseStatus.COMPLETED:
            break
        sleep(1)
    else:
        raise TimeoutError("SMPC did not finish in 100 tries.")

    # --------- Get SMPC result in globalnode -----------------
    sum_op_values_tableinfo = TableInfo.parse_raw(
        get_smpc_result_task_globalnode.delay(
            request_id=request_id,
            context_id=context_id,
            command_id="4",
            jobid=smpc_job_id,
        ).get()
    )

    # ----------------------- RUN GLOBAL UDF USING SMPC RESULTS ----------------------
    smpc_arg = NodeSMPCDTO(
        value=SMPCTablesInfo(
            template=globalnode_template_tableinfo,
            sum_op=sum_op_values_tableinfo,
        )
    )
    pos_args_str = NodeUDFPosArguments(args=[smpc_arg]).json()
    udf_results_str = run_udf_task_globalnode.delay(
        command_id="5",
        request_id=request_id,
        context_id=context_id,
        func_name=make_unique_func_name(smpc_global_step),
        positional_args_json=pos_args_str,
        keyword_args_json=NodeUDFKeyArguments(args={}).json(),
        use_smpc=True,
    ).get()

    global_step_result = NodeUDFResults.parse_raw(udf_results_str).results[0]
    assert isinstance(global_step_result, NodeTableDTO)

    expected_result = {"total_sum": input_table_1_name_sum + input_table_2_name_sum}
    validate_table_data_match_expected(
        db_cursor=globalnode_smpc_db_cursor,
        table_name=global_step_result.value.name,
        expected_values=expected_result,
    )
