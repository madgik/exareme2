import json
import uuid
from time import sleep
from typing import Any
from typing import Tuple

import pytest
import requests

from exareme2 import DType
from exareme2.algorithms.exareme2.udfgen import make_unique_func_name
from exareme2.smpc_cluster_communication import ADD_DATASET_ENDPOINT
from exareme2.smpc_cluster_communication import TRIGGER_COMPUTATION_ENDPOINT
from exareme2.smpc_cluster_communication import SMPCRequestData
from exareme2.smpc_cluster_communication import SMPCRequestType
from exareme2.smpc_cluster_communication import SMPCResponse
from exareme2.smpc_cluster_communication import SMPCResponseStatus
from exareme2.smpc_cluster_communication import get_smpc_result
from exareme2.worker_communication import ColumnInfo
from exareme2.worker_communication import SMPCTablesInfo
from exareme2.worker_communication import TableInfo
from exareme2.worker_communication import TableSchema
from exareme2.worker_communication import TableType
from exareme2.worker_communication import WorkerSMPCDTO
from exareme2.worker_communication import WorkerTableDTO
from exareme2.worker_communication import WorkerUDFKeyArguments
from exareme2.worker_communication import WorkerUDFPosArguments
from exareme2.worker_communication import WorkerUDFResults
from tests.algorithms.orphan_udfs import smpc_global_step
from tests.algorithms.orphan_udfs import smpc_local_step
from tests.standalone_tests.conftest import LOCALWORKER1_SMPC_CONFIG_FILE
from tests.standalone_tests.conftest import LOCALWORKER2_SMPC_CONFIG_FILE
from tests.standalone_tests.conftest import SMPC_COORDINATOR_ADDRESS
from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.conftest import create_table_in_db
from tests.standalone_tests.conftest import get_table_data_from_db
from tests.standalone_tests.conftest import get_worker_config_by_id
from tests.standalone_tests.conftest import insert_data_to_db
from tests.standalone_tests.workers_communication_helper import (
    get_celery_task_signature,
)

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
    localworker1_worker_service,
    use_localworker1_database,
    localworker1_celery_app,
    localworker1_db_cursor,
):
    localworker1_celery_app = localworker1_celery_app._celery_app
    run_udf_task = get_celery_task_signature("run_udf")

    input_table_info, input_table_name_sum = create_table_with_one_column_and_ten_rows(
        localworker1_db_cursor
    )

    pos_args_str = WorkerUDFPosArguments(
        args=[WorkerTableDTO(value=input_table_info)]
    ).json()
    udf_results_str = (
        localworker1_celery_app.signature(run_udf_task)
        .delay(
            command_id="1",
            request_id=request_id,
            context_id=context_id,
            func_name=make_unique_func_name(smpc_local_step),
            positional_args_json=pos_args_str,
            keyword_args_json=WorkerUDFKeyArguments(args={}).json(),
        )
        .get(timeout=TASKS_TIMEOUT)
    )

    results = WorkerUDFResults.parse_raw(udf_results_str).results
    assert len(results) == 1

    secure_transfer_result = results[0]
    assert isinstance(secure_transfer_result, WorkerTableDTO)

    expected_result = {
        "sum": {"data": input_table_name_sum, "operation": "sum", "type": "int"}
    }
    validate_table_data_match_expected(
        db_cursor=localworker1_db_cursor,
        table_name=secure_transfer_result.value.name,
        expected_values=expected_result,
    )


@pytest.mark.slow
def test_secure_transfer_input_with_smpc_off(
    localworker1_worker_service,
    use_localworker1_database,
    localworker1_celery_app,
    localworker1_db_cursor,
):
    localworker1_celery_app = localworker1_celery_app._celery_app
    run_udf_task = get_celery_task_signature("run_udf")

    (
        secure_transfer_results_tableinfo,
        secure_transfer_results_values_sum,
    ) = create_table_with_secure_transfer_results_with_smpc_off(localworker1_db_cursor)

    pos_args_str = WorkerUDFPosArguments(
        args=[WorkerTableDTO(value=secure_transfer_results_tableinfo)]
    ).json()

    udf_results_str = (
        localworker1_celery_app.signature(run_udf_task)
        .delay(
            command_id="1",
            request_id=request_id,
            context_id=context_id,
            func_name=make_unique_func_name(smpc_global_step),
            positional_args_json=pos_args_str,
            keyword_args_json=WorkerUDFKeyArguments(args={}).json(),
        )
        .get(timeout=TASKS_TIMEOUT)
    )

    results = WorkerUDFResults.parse_raw(udf_results_str).results
    assert len(results) == 1

    transfer_result = results[0]
    assert isinstance(transfer_result, WorkerTableDTO)

    expected_result = {"total_sum": secure_transfer_results_values_sum}
    validate_table_data_match_expected(
        db_cursor=localworker1_db_cursor,
        table_name=transfer_result.value.name,
        expected_values=expected_result,
    )


@pytest.mark.slow
@pytest.mark.very_slow
@pytest.mark.smpc
def test_validate_smpc_templates_match(
    smpc_localworker1_worker_service,
    use_smpc_localworker1_database,
    smpc_localworker1_celery_app,
    localworker1_smpc_db_cursor,
):
    smpc_localworker1_celery_app = smpc_localworker1_celery_app._celery_app
    validate_smpc_templates_match_task = get_celery_task_signature(
        "validate_smpc_templates_match"
    )

    table_info = create_table_with_multiple_secure_transfer_templates(
        localworker1_smpc_db_cursor, True
    )

    try:
        smpc_localworker1_celery_app.signature(
            validate_smpc_templates_match_task
        ).delay(request_id=request_id, table_name=table_info.name).get(
            timeout=TASKS_TIMEOUT
        )
    except Exception as exc:
        pytest.fail(f"No exception should be raised. Exception: {exc}")


@pytest.mark.slow
@pytest.mark.very_slow
@pytest.mark.smpc
def test_validate_smpc_templates_dont_match(
    smpc_localworker1_worker_service,
    use_smpc_localworker1_database,
    smpc_localworker1_celery_app,
    localworker1_smpc_db_cursor,
):
    smpc_localworker1_celery_app = smpc_localworker1_celery_app._celery_app
    validate_smpc_templates_match_task = get_celery_task_signature(
        "validate_smpc_templates_match"
    )

    table_info = create_table_with_multiple_secure_transfer_templates(
        localworker1_smpc_db_cursor, False
    )

    with pytest.raises(ValueError) as exc:
        smpc_localworker1_celery_app.signature(
            validate_smpc_templates_match_task
        ).delay(request_id=request_id, table_name=table_info.name).get(
            timeout=TASKS_TIMEOUT
        )
    assert "SMPC templates dont match." in str(exc)


@pytest.mark.slow
@pytest.mark.very_slow
@pytest.mark.smpc
def test_secure_transfer_run_udf_flow_with_smpc_on(
    smpc_localworker1_worker_service,
    use_smpc_localworker1_database,
    smpc_localworker1_celery_app,
    localworker1_smpc_db_cursor,
):
    smpc_localworker1_celery_app = smpc_localworker1_celery_app._celery_app
    run_udf_task = get_celery_task_signature("run_udf")

    # ----------------------- SECURE TRANSFER OUTPUT ----------------------
    input_table_name, input_table_name_sum = create_table_with_one_column_and_ten_rows(
        localworker1_smpc_db_cursor
    )

    pos_args_str = WorkerUDFPosArguments(
        args=[WorkerTableDTO(value=input_table_name)]
    ).json()

    udf_results_str = (
        smpc_localworker1_celery_app.signature(run_udf_task)
        .delay(
            command_id="1",
            request_id=request_id,
            context_id=context_id,
            func_name=make_unique_func_name(smpc_local_step),
            positional_args_json=pos_args_str,
            keyword_args_json=WorkerUDFKeyArguments(args={}).json(),
            use_smpc=True,
        )
        .get(timeout=TASKS_TIMEOUT)
    )

    local_step_results = WorkerUDFResults.parse_raw(udf_results_str).results
    assert len(local_step_results) == 1

    smpc_result = local_step_results[0]
    assert isinstance(smpc_result, WorkerSMPCDTO)

    assert smpc_result.value.template is not None
    expected_template = {"sum": {"data": 0, "operation": "sum", "type": "int"}}
    validate_table_data_match_expected(
        db_cursor=localworker1_smpc_db_cursor,
        table_name=smpc_result.value.template.name,
        expected_values=expected_template,
    )

    assert smpc_result.value.sum_op is not None
    expected_sum_op_values = [input_table_name_sum]
    validate_table_data_match_expected(
        db_cursor=localworker1_smpc_db_cursor,
        table_name=smpc_result.value.sum_op.name,
        expected_values=expected_sum_op_values,
    )

    # ----------------------- SECURE TRANSFER INPUT----------------------
    pos_args_str = WorkerUDFPosArguments(args=[smpc_result]).json()

    udf_results_str = (
        smpc_localworker1_celery_app.signature(run_udf_task)
        .delay(
            command_id="2",
            request_id=request_id,
            context_id=context_id,
            func_name=make_unique_func_name(smpc_global_step),
            positional_args_json=pos_args_str,
            keyword_args_json=WorkerUDFKeyArguments(args={}).json(),
            use_smpc=True,
        )
        .get(timeout=TASKS_TIMEOUT)
    )

    global_step_results = WorkerUDFResults.parse_raw(udf_results_str).results
    assert len(global_step_results) == 1

    global_step_result = global_step_results[0]
    assert isinstance(global_step_result, WorkerTableDTO)

    expected_result = {"total_sum": input_table_name_sum}
    validate_table_data_match_expected(
        db_cursor=localworker1_smpc_db_cursor,
        table_name=global_step_result.value.name,
        expected_values=expected_result,
    )


@pytest.mark.slow
@pytest.mark.very_slow
@pytest.mark.smpc
def test_load_data_to_smpc_client_from_globalworker_fails(
    smpc_globalworker_worker_service,
    smpc_globalworker_celery_app,
):
    smpc_globalworker_celery_app = smpc_globalworker_celery_app._celery_app
    load_data_to_smpc_client_task = get_celery_task_signature(
        "load_data_to_smpc_client"
    )

    with pytest.raises(PermissionError) as exc:
        smpc_globalworker_celery_app.signature(load_data_to_smpc_client_task).delay(
            request_id=request_id,
            table_name="whatever",
            jobid="whatever",
        ).get(timeout=TASKS_TIMEOUT)
    assert "load_data_to_smpc_client is allowed only for a LOCALWORKER." in str(exc)


@pytest.mark.slow
@pytest.mark.very_slow
@pytest.mark.smpc
@pytest.mark.smpc_cluster
def test_load_data_to_smpc_client(
    smpc_localworker1_worker_service,
    use_smpc_localworker1_database,
    smpc_localworker1_celery_app,
    localworker1_smpc_db_cursor,
    smpc_cluster,
):
    smpc_localworker1_celery_app = smpc_localworker1_celery_app._celery_app
    table_info, sum_op_values_str = create_table_with_smpc_sum_op_values(
        localworker1_smpc_db_cursor
    )
    load_data_to_smpc_client_task = get_celery_task_signature(
        "load_data_to_smpc_client"
    )

    smpc_localworker1_celery_app.signature(load_data_to_smpc_client_task).delay(
        request_id=request_id,
        table_name=table_info.name,
        jobid=smpc_job_id,
    ).get(timeout=TASKS_TIMEOUT)

    worker_config = get_worker_config_by_id(LOCALWORKER1_SMPC_CONFIG_FILE)
    request_url = (
        worker_config.smpc.client_address + SMPC_GET_DATASET_ENDPOINT + smpc_job_id
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
def test_get_smpc_result_from_localworker_fails(
    smpc_localworker1_worker_service,
    smpc_localworker1_celery_app,
):
    smpc_localworker1_celery_app = smpc_localworker1_celery_app._celery_app
    get_smpc_result_task = get_celery_task_signature("get_smpc_result")

    with pytest.raises(PermissionError) as exc:
        smpc_localworker1_celery_app.signature(get_smpc_result_task).delay(
            request_id="whatever",
            context_id="whatever",
            command_id="whatever",
            jobid="whatever",
        ).get(timeout=TASKS_TIMEOUT)
    assert "get_smpc_result is allowed only for a GLOBALWORKER." in str(exc)


@pytest.mark.slow
@pytest.mark.very_slow
@pytest.mark.smpc
@pytest.mark.smpc_cluster
def test_get_smpc_result(
    smpc_globalworker_worker_service,
    use_smpc_globalworker_database,
    smpc_globalworker_celery_app,
    globalworker_smpc_db_cursor,
    smpc_cluster,
):
    smpc_globalworker_celery_app = smpc_globalworker_celery_app._celery_app
    get_smpc_result_task = get_celery_task_signature("get_smpc_result")

    # --------------- LOAD Dataset to SMPC --------------------
    worker_config = get_worker_config_by_id(LOCALWORKER1_SMPC_CONFIG_FILE)
    request_url = worker_config.smpc.client_address + ADD_DATASET_ENDPOINT + smpc_job_id
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
        {"computationType": "sum", "clients": [worker_config.smpc.client_id]}
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

    # --------------- GET SMPC RESULT IN GLOBALWORKER ------------------------
    result_tableinfo = TableInfo.parse_raw(
        smpc_globalworker_celery_app.signature(get_smpc_result_task)
        .delay(
            request_id=request_id,
            context_id=context_id,
            command_id=command_id,
            jobid=smpc_job_id,
        )
        .get(timeout=TASKS_TIMEOUT)
    )

    validate_table_data_match_expected(
        db_cursor=globalworker_smpc_db_cursor,
        table_name=result_tableinfo.name,
        expected_values=smpc_computation_data,
    )


@pytest.mark.slow
@pytest.mark.very_slow
@pytest.mark.smpc
@pytest.mark.smpc_cluster
def test_orchestrate_SMPC_between_two_localworkers_and_the_globalworker(
    smpc_globalworker_worker_service,
    smpc_localworker1_worker_service,
    smpc_localworker2_worker_service,
    use_smpc_globalworker_database,
    use_smpc_localworker1_database,
    use_smpc_localworker2_database,
    smpc_globalworker_celery_app,
    smpc_localworker1_celery_app,
    smpc_localworker2_celery_app,
    localworker1_smpc_db_cursor,
    localworker2_smpc_db_cursor,
    globalworker_smpc_db_cursor,
    smpc_cluster,
):
    smpc_globalworker_celery_app = smpc_globalworker_celery_app._celery_app
    smpc_localworker1_celery_app = smpc_localworker1_celery_app._celery_app
    smpc_localworker2_celery_app = smpc_localworker2_celery_app._celery_app

    run_udf_task_globalworker = smpc_globalworker_celery_app.signature(
        get_celery_task_signature("run_udf")
    )
    run_udf_task_localworker1 = smpc_localworker1_celery_app.signature(
        get_celery_task_signature("run_udf")
    )
    run_udf_task_localworker2 = smpc_localworker2_celery_app.signature(
        get_celery_task_signature("run_udf")
    )
    create_remote_table_task_globalworker = smpc_globalworker_celery_app.signature(
        get_celery_task_signature("create_remote_table")
    )
    create_merge_table_task_globalworker = smpc_globalworker_celery_app.signature(
        get_celery_task_signature("create_merge_table")
    )
    validate_smpc_templates_match_task_globalworker = (
        smpc_globalworker_celery_app.signature(
            get_celery_task_signature("validate_smpc_templates_match")
        )
    )
    load_data_to_smpc_client_task_localworker1 = smpc_localworker1_celery_app.signature(
        get_celery_task_signature("load_data_to_smpc_client")
    )
    load_data_to_smpc_client_task_localworker2 = smpc_localworker2_celery_app.signature(
        get_celery_task_signature("load_data_to_smpc_client")
    )
    get_smpc_result_task_globalworker = smpc_globalworker_celery_app.signature(
        get_celery_task_signature("get_smpc_result")
    )

    # ---------------- CREATE LOCAL TABLES WITH INITIAL DATA ----------------------
    (
        input_table_1_name,
        input_table_1_name_sum,
    ) = create_table_with_one_column_and_ten_rows(localworker1_smpc_db_cursor)
    (
        input_table_2_name,
        input_table_2_name_sum,
    ) = create_table_with_one_column_and_ten_rows(localworker2_smpc_db_cursor)

    # ---------------- RUN LOCAL UDFS WITH SECURE TRANSFER OUTPUT ----------------------
    pos_args_str_localworker1 = WorkerUDFPosArguments(
        args=[WorkerTableDTO(value=input_table_1_name)]
    ).json()
    pos_args_str_localworker2 = WorkerUDFPosArguments(
        args=[WorkerTableDTO(value=input_table_2_name)]
    ).json()

    udf_results_str_localworker1 = run_udf_task_localworker1.delay(
        command_id="1",
        request_id=request_id,
        context_id=context_id,
        func_name=make_unique_func_name(smpc_local_step),
        positional_args_json=pos_args_str_localworker1,
        keyword_args_json=WorkerUDFKeyArguments(args={}).json(),
        use_smpc=True,
    ).get()

    udf_results_str_localworker2 = run_udf_task_localworker2.delay(
        command_id="2",
        request_id=request_id,
        context_id=context_id,
        func_name=make_unique_func_name(smpc_local_step),
        positional_args_json=pos_args_str_localworker2,
        keyword_args_json=WorkerUDFKeyArguments(args={}).json(),
        use_smpc=True,
    ).get()

    local_1_smpc_result = WorkerUDFResults.parse_raw(
        udf_results_str_localworker1
    ).results[0]
    assert isinstance(local_1_smpc_result, WorkerSMPCDTO)
    local_2_smpc_result = WorkerUDFResults.parse_raw(
        udf_results_str_localworker2
    ).results[0]
    assert isinstance(local_2_smpc_result, WorkerSMPCDTO)

    # ---------- CREATE REMOTE/MERGE TABLE ON GLOBALWORKER WITH SMPC TEMPLATE ---------
    localworker1_config = get_worker_config_by_id(LOCALWORKER1_SMPC_CONFIG_FILE)
    localworker2_config = get_worker_config_by_id(LOCALWORKER2_SMPC_CONFIG_FILE)

    localworker_1_monetdb_sock_address = (
        f"{str(localworker1_config.monetdb.ip)}:{localworker1_config.monetdb.port}"
    )
    localworker_2_monetdb_sock_address = (
        f"{str(localworker2_config.monetdb.ip)}:{localworker2_config.monetdb.port}"
    )
    create_remote_table_task_globalworker.delay(
        request_id=request_id,
        table_name=local_1_smpc_result.value.template.name,
        table_schema_json=local_1_smpc_result.value.template.schema_.json(),
        monetdb_socket_address=localworker_1_monetdb_sock_address,
    ).get()
    create_remote_table_task_globalworker.delay(
        request_id=request_id,
        table_name=local_2_smpc_result.value.template.name,
        table_schema_json=local_2_smpc_result.value.template.schema_.json(),
        monetdb_socket_address=localworker_2_monetdb_sock_address,
    ).get()
    globalworker_template_tableinfo = TableInfo.parse_raw(
        create_merge_table_task_globalworker.delay(
            request_id=request_id,
            context_id=context_id,
            command_id="3",
            table_infos_json=[
                local_1_smpc_result.value.template.json(),
                local_2_smpc_result.value.template.json(),
            ],
        ).get()
    )

    validate_smpc_templates_match_task_globalworker.delay(
        request_id=request_id,
        table_name=globalworker_template_tableinfo.name,
    ).get()

    # --------- LOAD LOCALWORKER ADD OP DATA TO SMPC CLIENTS -----------------
    smpc_client_1 = load_data_to_smpc_client_task_localworker1.delay(
        request_id=request_id,
        table_name=local_1_smpc_result.value.sum_op.name,
        jobid=smpc_job_id,
    ).get()
    smpc_client_2 = load_data_to_smpc_client_task_localworker2.delay(
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

    # --------- Get SMPC result in globalworker -----------------
    sum_op_values_tableinfo = TableInfo.parse_raw(
        get_smpc_result_task_globalworker.delay(
            request_id=request_id,
            context_id=context_id,
            command_id="4",
            jobid=smpc_job_id,
        ).get()
    )

    # ----------------------- RUN GLOBAL UDF USING SMPC RESULTS ----------------------
    smpc_arg = WorkerSMPCDTO(
        value=SMPCTablesInfo(
            template=globalworker_template_tableinfo,
            sum_op=sum_op_values_tableinfo,
        )
    )
    pos_args_str = WorkerUDFPosArguments(args=[smpc_arg]).json()
    udf_results_str = run_udf_task_globalworker.delay(
        command_id="5",
        request_id=request_id,
        context_id=context_id,
        func_name=make_unique_func_name(smpc_global_step),
        positional_args_json=pos_args_str,
        keyword_args_json=WorkerUDFKeyArguments(args={}).json(),
        use_smpc=True,
    ).get()

    global_step_result = WorkerUDFResults.parse_raw(udf_results_str).results[0]
    assert isinstance(global_step_result, WorkerTableDTO)

    expected_result = {"total_sum": input_table_1_name_sum + input_table_2_name_sum}
    validate_table_data_match_expected(
        db_cursor=globalworker_smpc_db_cursor,
        table_name=global_step_result.value.name,
        expected_values=expected_result,
    )
