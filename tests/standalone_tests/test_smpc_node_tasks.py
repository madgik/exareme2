import json
import uuid
from typing import Tuple

import pytest
import requests

from mipengine import DType
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import NodeSMPCDTO
from mipengine.node_tasks_DTOs import NodeSMPCValueDTO
from mipengine.node_tasks_DTOs import NodeTableDTO
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import UDFKeyArguments
from mipengine.node_tasks_DTOs import UDFPosArguments
from mipengine.node_tasks_DTOs import UDFResults
from mipengine.smpc_cluster_comm_helpers import ADD_DATASET_ENDPOINT
from mipengine.smpc_cluster_comm_helpers import TRIGGER_COMPUTATION_ENDPOINT
from mipengine.smpc_DTOs import SMPCRequestData
from mipengine.smpc_DTOs import SMPCRequestType
from mipengine.udfgen import make_unique_func_name
from tests.algorithms.orphan_udfs import smpc_global_step
from tests.algorithms.orphan_udfs import smpc_local_step
from tests.standalone_tests.conftest import GLOBALNODE_SMPC_CONFIG_FILE
from tests.standalone_tests.conftest import LOCALNODE1_CONFIG_FILE
from tests.standalone_tests.conftest import LOCALNODE1_SMPC_CONFIG_FILE
from tests.standalone_tests.conftest import LOCALNODE2_SMPC_CONFIG_FILE
from tests.standalone_tests.nodes_communication_helper import get_celery_app
from tests.standalone_tests.nodes_communication_helper import get_celery_task_signature
from tests.standalone_tests.nodes_communication_helper import get_node_config_by_id
from tests.standalone_tests.test_udfs import create_table_with_one_column_and_ten_rows

request_id = "testsmpcudfs" + str(uuid.uuid4().hex)[:10] + "request"
context_id = "testsmpcudfs" + str(uuid.uuid4().hex)[:10]
command_id = "command123"
smpc_job_id = "testKey123"
SMPC_GET_DATASET_ENDPOINT = "/api/update-dataset/"
SMPC_COORDINATOR_ADDRESS = "http://172.17.0.1:12314"


@pytest.fixture(scope="session")
def localnode_1_celery_app():
    config = get_node_config_by_id(LOCALNODE1_CONFIG_FILE)
    yield get_celery_app(config)


@pytest.fixture(scope="session")
def smpc_globalnode_celery_app():
    config = get_node_config_by_id(GLOBALNODE_SMPC_CONFIG_FILE)
    yield get_celery_app(config)


@pytest.fixture(scope="session")
def smpc_localnode1_celery_app():
    config = get_node_config_by_id(LOCALNODE1_SMPC_CONFIG_FILE)
    yield get_celery_app(config)


@pytest.fixture(scope="session")
def smpc_localnode2_celery_app():
    config = get_node_config_by_id(LOCALNODE2_SMPC_CONFIG_FILE)
    yield get_celery_app(config)


def create_secure_transfer_table(celery_app) -> str:
    create_table_task = get_celery_task_signature(celery_app, "create_table")

    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="node_id", dtype=DType.STR),
            ColumnInfo(name="secure_transfer", dtype=DType.JSON),
        ]
    )
    table_name = create_table_task.delay(
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    ).get()

    return table_name


# TODO More dynamic so it can receive any secure_transfer values
def create_table_with_secure_transfer_results_with_smpc_off(
    celery_app,
) -> Tuple[str, int]:
    insert_data_to_table_task = get_celery_task_signature(
        celery_app, "insert_data_to_table"
    )

    table_name = create_secure_transfer_table(celery_app)

    secure_transfer_1_value = 100
    secure_transfer_2_value = 11

    secure_transfer_1 = {"sum": {"data": secure_transfer_1_value, "operation": "sum"}}
    secure_transfer_2 = {"sum": {"data": secure_transfer_2_value, "operation": "sum"}}
    values = [
        ["localnode1", json.dumps(secure_transfer_1)],
        ["localnode2", json.dumps(secure_transfer_2)],
    ]
    insert_data_to_table_task.delay(
        request_id=request_id, table_name=table_name, values=values
    ).get()

    return table_name, secure_transfer_1_value + secure_transfer_2_value


def create_table_with_multiple_secure_transfer_templates(
    celery_app, similar: bool
) -> str:
    insert_data_to_table_task = get_celery_task_signature(
        celery_app, "insert_data_to_table"
    )

    table_name = create_secure_transfer_table(celery_app)

    secure_transfer_template = {"sum": {"data": [0, 1, 2, 3], "operation": "sum"}}
    different_secure_transfer_template = {"sum": {"data": 0, "operation": "sum"}}

    if similar:
        values = [
            ["localnode1", json.dumps(secure_transfer_template)],
            ["localnode2", json.dumps(secure_transfer_template)],
        ]
    else:
        values = [
            ["localnode1", json.dumps(secure_transfer_template)],
            ["localnode2", json.dumps(different_secure_transfer_template)],
        ]

    insert_data_to_table_task.delay(
        request_id=request_id, table_name=table_name, values=values
    ).get()

    return table_name


def create_table_with_smpc_sum_op_values(celery_app) -> Tuple[str, str]:
    insert_data_to_table_task = get_celery_task_signature(
        celery_app, "insert_data_to_table"
    )
    table_name = create_secure_transfer_table(celery_app)

    sum_op_values = [0, 1, 2, 3, 4, 5]
    values = [
        ["localnode1", json.dumps(sum_op_values)],
    ]

    insert_data_to_table_task.delay(
        request_id=request_id, table_name=table_name, values=values
    ).get()

    return table_name, json.dumps(sum_op_values)


def validate_dict_table_data_match_expected(
    get_table_data_task, table_name, expected_values
):
    assert table_name is not None
    table_data_str = get_table_data_task.delay(
        request_id=request_id, table_name=table_name
    ).get()
    table_data: TableData = TableData.parse_raw(table_data_str)
    result_str, *_ = table_data.columns[1].data
    result = json.loads(result_str)
    assert result == expected_values


def test_secure_transfer_output_with_smpc_off(
    localnode1_node_service, use_localnode1_database, localnode_1_celery_app
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
        request_id=request_id,
        context_id=context_id,
        func_name=make_unique_func_name(smpc_local_step),
        positional_args_json=pos_args_str,
        keyword_args_json=UDFKeyArguments(args={}).json(),
    ).get()

    results = UDFResults.parse_raw(udf_results_str).results
    assert len(results) == 1

    secure_transfer_result = results[0]
    assert isinstance(secure_transfer_result, NodeTableDTO)

    expected_result = {"sum": {"data": input_table_name_sum, "operation": "sum"}}
    validate_dict_table_data_match_expected(
        get_table_data_task,
        secure_transfer_result.value,
        expected_result,
    )


def test_secure_transfer_input_with_smpc_off(
    localnode1_node_service, use_localnode1_database, localnode_1_celery_app
):
    run_udf_task = get_celery_task_signature(localnode_1_celery_app, "run_udf")
    get_table_data_task = get_celery_task_signature(
        localnode_1_celery_app, "get_table_data"
    )

    (
        secure_transfer_results_tablename,
        secure_transfer_results_values_sum,
    ) = create_table_with_secure_transfer_results_with_smpc_off(localnode_1_celery_app)

    pos_args_str = UDFPosArguments(
        args=[NodeTableDTO(value=secure_transfer_results_tablename)]
    ).json()

    udf_results_str = run_udf_task.delay(
        command_id="1",
        request_id=request_id,
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


def test_validate_smpc_templates_match(
    smpc_localnode1_node_service,
    use_smpc_localnode1_database,
    smpc_localnode1_celery_app,
):
    validate_smpc_templates_match_task = get_celery_task_signature(
        smpc_localnode1_celery_app, "validate_smpc_templates_match"
    )

    table_name = create_table_with_multiple_secure_transfer_templates(
        smpc_localnode1_celery_app, True
    )

    try:
        validate_smpc_templates_match_task.delay(
            request_id=request_id, table_name=table_name
        ).get()
    except Exception as exc:
        pytest.fail(f"No exception should be raised. Exception: {exc}")


def test_validate_smpc_templates_dont_match(
    smpc_localnode1_node_service,
    use_smpc_localnode1_database,
    smpc_localnode1_celery_app,
):
    validate_smpc_templates_match_task = get_celery_task_signature(
        smpc_localnode1_celery_app, "validate_smpc_templates_match"
    )

    table_name = create_table_with_multiple_secure_transfer_templates(
        smpc_localnode1_celery_app, False
    )

    with pytest.raises(ValueError) as exc:
        validate_smpc_templates_match_task.delay(
            request_id=request_id, table_name=table_name
        ).get()
    assert "SMPC templates dont match." in str(exc)


def test_secure_transfer_run_udf_flow_with_smpc_on(
    smpc_localnode1_node_service,
    use_smpc_localnode1_database,
    smpc_localnode1_celery_app,
):
    run_udf_task = get_celery_task_signature(smpc_localnode1_celery_app, "run_udf")
    get_table_data_task = get_celery_task_signature(
        smpc_localnode1_celery_app, "get_table_data"
    )

    # ----------------------- SECURE TRANSFER OUTPUT ----------------------

    input_table_name, input_table_name_sum = create_table_with_one_column_and_ten_rows(
        smpc_localnode1_celery_app
    )

    pos_args_str = UDFPosArguments(args=[NodeTableDTO(value=input_table_name)]).json()

    udf_results_str = run_udf_task.delay(
        command_id="1",
        request_id=request_id,
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
    expected_template = {"sum": {"data": 0, "operation": "sum"}}
    validate_dict_table_data_match_expected(
        get_table_data_task,
        smpc_result.value.template.value,
        expected_template,
    )

    assert smpc_result.value.sum_op_values is not None
    expected_sum_op_values = [input_table_name_sum]
    validate_dict_table_data_match_expected(
        get_table_data_task,
        smpc_result.value.sum_op_values.value,
        expected_sum_op_values,
    )

    # ----------------------- SECURE TRANSFER INPUT----------------------

    # Providing as input the smpc_result created from the previous udf (local step)
    smpc_arg = NodeSMPCDTO(
        value=NodeSMPCValueDTO(
            template=NodeTableDTO(value=smpc_result.value.template.value),
            sum_op_values=NodeTableDTO(value=smpc_result.value.sum_op_values.value),
        )
    )

    pos_args_str = UDFPosArguments(args=[smpc_arg]).json()

    udf_results_str = run_udf_task.delay(
        command_id="2",
        request_id=request_id,
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


def test_load_data_to_smpc_client_from_globalnode_fails(
    smpc_globalnode_node_service,
    smpc_globalnode_celery_app,
):
    load_data_to_smpc_client_task = get_celery_task_signature(
        smpc_globalnode_celery_app, "load_data_to_smpc_client"
    )

    with pytest.raises(PermissionError) as exc:
        load_data_to_smpc_client_task.delay(
            request_id=request_id,
            table_name="whatever",
            jobid="whatever",
        ).get()
    assert "load_data_to_smpc_client is allowed only for a LOCALNODE." in str(exc)


@pytest.mark.smpc
def test_load_data_to_smpc_client(
    smpc_localnode1_node_service,
    use_smpc_localnode1_database,
    smpc_localnode1_celery_app,
    smpc_cluster,
):
    table_name, sum_op_values_str = create_table_with_smpc_sum_op_values(
        smpc_localnode1_celery_app
    )

    load_data_to_smpc_client_task = get_celery_task_signature(
        smpc_localnode1_celery_app, "load_data_to_smpc_client"
    )

    load_data_to_smpc_client_task.delay(
        request_id=request_id,
        table_name=table_name,
        jobid=smpc_job_id,
    ).get()

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

    # TODO Remove when smpc cluster call is fixed
    # The problem is that the call returns the integers as string
    # assert response.text == sum_op_values_str
    result = json.loads(response.text)
    result = [int(elem) for elem in result]
    assert json.dumps(result) == sum_op_values_str


def test_get_smpc_result_from_localnode_fails(
    smpc_localnode1_node_service,
    smpc_localnode1_celery_app,
):
    get_smpc_result_task = get_celery_task_signature(
        smpc_localnode1_celery_app, "get_smpc_result"
    )

    with pytest.raises(PermissionError) as exc:
        get_smpc_result_task.delay(
            request_id="whatever",
            context_id="whatever",
            command_id="whatever",
            jobid="whatever",
        ).get()
    assert "get_smpc_result is allowed only for a GLOBALNODE." in str(exc)


@pytest.mark.smpc
def test_get_smpc_result(
    smpc_globalnode_node_service,
    use_smpc_globalnode_database,
    smpc_globalnode_celery_app,
    smpc_cluster,
):
    get_smpc_result_task = get_celery_task_signature(
        smpc_globalnode_celery_app, "get_smpc_result"
    )
    get_table_data_task = get_celery_task_signature(
        smpc_globalnode_celery_app, "get_table_data"
    )

    # --------------- LOAD Dataset to SMPC --------------------
    node_config = get_node_config_by_id(LOCALNODE1_SMPC_CONFIG_FILE)
    request_url = node_config.smpc.client_address + ADD_DATASET_ENDPOINT + smpc_job_id
    request_headers = {"Content-type": "application/json", "Accept": "text/plain"}
    smpc_computation_data = [100]
    response = requests.post(
        request_url,
        data=json.dumps(smpc_computation_data),
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

    # --------------- GET SMPC RESULT IN GLOBALNODE ------------------------
    result_tablename = get_smpc_result_task.delay(
        request_id=request_id,
        context_id=context_id,
        command_id=command_id,
        jobid=smpc_job_id,
    ).get()

    validate_dict_table_data_match_expected(
        get_table_data_task, result_tablename, smpc_computation_data
    )


@pytest.mark.smpc
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
    smpc_cluster,
):
    run_udf_task_globalnode = get_celery_task_signature(
        smpc_globalnode_celery_app, "run_udf"
    )
    run_udf_task_localnode1 = get_celery_task_signature(
        smpc_localnode1_celery_app, "run_udf"
    )
    run_udf_task_localnode2 = get_celery_task_signature(
        smpc_localnode2_celery_app, "run_udf"
    )
    get_table_schema_task_localnode1 = get_celery_task_signature(
        smpc_localnode1_celery_app, "get_table_schema"
    )
    create_remote_table_task_globalnode = get_celery_task_signature(
        smpc_globalnode_celery_app, "create_remote_table"
    )
    create_merge_table_task_globalnode = get_celery_task_signature(
        smpc_globalnode_celery_app, "create_merge_table"
    )
    get_table_data_task_globalnode = get_celery_task_signature(
        smpc_globalnode_celery_app, "get_table_data"
    )
    validate_smpc_templates_match_task_globalnode = get_celery_task_signature(
        smpc_globalnode_celery_app, "validate_smpc_templates_match"
    )
    load_data_to_smpc_client_task_localnode1 = get_celery_task_signature(
        smpc_localnode1_celery_app, "load_data_to_smpc_client"
    )
    load_data_to_smpc_client_task_localnode2 = get_celery_task_signature(
        smpc_localnode2_celery_app, "load_data_to_smpc_client"
    )
    get_smpc_result_task_globalnode = get_celery_task_signature(
        smpc_globalnode_celery_app, "get_smpc_result"
    )

    # ---------------- CREATE LOCAL TABLES WITH INITIAL DATA ----------------------
    (
        input_table_1_name,
        input_table_1_name_sum,
    ) = create_table_with_one_column_and_ten_rows(smpc_localnode1_celery_app)
    (
        input_table_2_name,
        input_table_2_name_sum,
    ) = create_table_with_one_column_and_ten_rows(smpc_localnode2_celery_app)

    # ---------------- RUN LOCAL UDFS WITH SECURE TRANSFER OUTPUT ----------------------
    pos_args_str_localnode1 = UDFPosArguments(
        args=[NodeTableDTO(value=input_table_1_name)]
    ).json()
    pos_args_str_localnode2 = UDFPosArguments(
        args=[NodeTableDTO(value=input_table_2_name)]
    ).json()

    udf_results_str_localnode1 = run_udf_task_localnode1.delay(
        command_id="1",
        request_id=request_id,
        context_id=context_id,
        func_name=make_unique_func_name(smpc_local_step),
        positional_args_json=pos_args_str_localnode1,
        keyword_args_json=UDFKeyArguments(args={}).json(),
        use_smpc=True,
    ).get()

    udf_results_str_localnode2 = run_udf_task_localnode2.delay(
        command_id="2",
        request_id=request_id,
        context_id=context_id,
        func_name=make_unique_func_name(smpc_local_step),
        positional_args_json=pos_args_str_localnode2,
        keyword_args_json=UDFKeyArguments(args={}).json(),
        use_smpc=True,
    ).get()

    local_1_smpc_result = UDFResults.parse_raw(udf_results_str_localnode1).results[0]
    assert isinstance(local_1_smpc_result, NodeSMPCDTO)
    local_2_smpc_result = UDFResults.parse_raw(udf_results_str_localnode2).results[0]
    assert isinstance(local_2_smpc_result, NodeSMPCDTO)

    # ---------- CREATE REMOTE/MERGE TABLE ON GLOBALNODE WITH SMPC TEMPLATE ---------
    template_table_schema_str = get_table_schema_task_localnode1.delay(
        request_id=request_id, table_name=local_1_smpc_result.value.template.value
    ).get()

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
        table_name=local_1_smpc_result.value.template.value,
        table_schema_json=template_table_schema_str,
        monetdb_socket_address=localnode_1_monetdb_sock_address,
    ).get()
    create_remote_table_task_globalnode.delay(
        request_id=request_id,
        table_name=local_2_smpc_result.value.template.value,
        table_schema_json=template_table_schema_str,
        monetdb_socket_address=localnode_2_monetdb_sock_address,
    ).get()
    globalnode_template_tablename = create_merge_table_task_globalnode.delay(
        request_id=request_id,
        context_id=context_id,
        command_id="3",
        table_names=[
            local_1_smpc_result.value.template.value,
            local_2_smpc_result.value.template.value,
        ],
    ).get()

    validate_smpc_templates_match_task_globalnode.delay(
        request_id=request_id,
        table_name=globalnode_template_tablename,
    ).get()

    # --------- LOAD LOCALNODE ADD OP DATA TO SMPC CLIENTS -----------------
    smpc_client_1 = load_data_to_smpc_client_task_localnode1.delay(
        request_id=request_id,
        table_name=local_1_smpc_result.value.sum_op_values.value,
        jobid=smpc_job_id,
    ).get()
    smpc_client_2 = load_data_to_smpc_client_task_localnode2.delay(
        request_id=request_id,
        table_name=local_2_smpc_result.value.sum_op_values.value,
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

    # --------- Get Results of SMPC in globalnode -----------------
    sum_op_values_tablename = get_smpc_result_task_globalnode.delay(
        request_id=request_id,
        context_id=context_id,
        command_id="4",
        jobid=smpc_job_id,
    ).get()

    # ----------------------- RUN GLOBAL UDF USING SMPC RESULTS ----------------------
    smpc_arg = NodeSMPCDTO(
        value=NodeSMPCValueDTO(
            template=NodeTableDTO(value=globalnode_template_tablename),
            sum_op_values=NodeTableDTO(value=sum_op_values_tablename),
        )
    )
    pos_args_str = UDFPosArguments(args=[smpc_arg]).json()
    udf_results_str = run_udf_task_globalnode.delay(
        command_id="5",
        request_id=request_id,
        context_id=context_id,
        func_name=make_unique_func_name(smpc_global_step),
        positional_args_json=pos_args_str,
        keyword_args_json=UDFKeyArguments(args={}).json(),
        use_smpc=True,
    ).get()

    global_step_result = UDFResults.parse_raw(udf_results_str).results[0]
    assert isinstance(global_step_result, NodeTableDTO)

    expected_result = {"total_sum": input_table_1_name_sum + input_table_2_name_sum}
    validate_dict_table_data_match_expected(
        get_table_data_task_globalnode,
        global_step_result.value,
        expected_result,
    )
