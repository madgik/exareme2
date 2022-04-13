from logging import Logger
from time import sleep
from typing import List
from typing import Optional
from typing import Tuple

from mipengine import smpc_cluster_comm_helpers as smpc_cluster
from mipengine.controller import config as ctrl_config
from mipengine.controller.algorithm_executor_node_data_objects import TableName
from mipengine.controller.algorithm_executor_nodes import GlobalNode
from mipengine.controller.algorithm_flow_data_objects import LocalNodesSMPCTables
from mipengine.controller.algorithm_flow_data_objects import LocalNodesTable
from mipengine.smpc_cluster_comm_helpers import SMPCComputationError
from mipengine.smpc_cluster_comm_helpers import trigger_smpc
from mipengine.smpc_DTOs import SMPCRequestType
from mipengine.smpc_DTOs import SMPCResponse
from mipengine.smpc_DTOs import SMPCResponseStatus


def get_smpc_job_id(
    context_id: str, command_id: int, operation: SMPCRequestType
) -> str:
    return context_id + "_" + str(command_id) + "_" + str(operation)


def load_operation_data_to_smpc_clients(
    command_id: int,
    local_nodes_table: Optional[LocalNodesTable],
    op_type: SMPCRequestType,
) -> List[str]:
    smpc_clients = []
    if local_nodes_table:
        for node, table in local_nodes_table.nodes_tables.items():
            smpc_clients.append(
                node.load_data_to_smpc_client(
                    table_name=table.full_table_name,
                    jobid=get_smpc_job_id(table.context_id, command_id, op_type),
                )
            )
    return smpc_clients


def load_data_to_smpc_clients(
    command_id: int, smpc_tables: LocalNodesSMPCTables
) -> Tuple[List[str], List[str], List[str], List[str]]:
    sum_op_smpc_clients = load_operation_data_to_smpc_clients(
        command_id, smpc_tables.sum_op_local_nodes_table, SMPCRequestType.SUM
    )
    min_op_smpc_clients = load_operation_data_to_smpc_clients(
        command_id, smpc_tables.min_op_local_nodes_table, SMPCRequestType.MIN
    )
    max_op_smpc_clients = load_operation_data_to_smpc_clients(
        command_id, smpc_tables.max_op_local_nodes_table, SMPCRequestType.MAX
    )
    union_op_smpc_clients = load_operation_data_to_smpc_clients(
        command_id, smpc_tables.union_op_local_nodes_table, SMPCRequestType.UNION
    )
    return (
        sum_op_smpc_clients,
        min_op_smpc_clients,
        max_op_smpc_clients,
        union_op_smpc_clients,
    )


def trigger_smpc_operation(
    logger: Logger,
    context_id: str,
    command_id: int,
    op_type: SMPCRequestType,
    smpc_op_clients: List[str],
) -> bool:
    trigger_smpc(
        logger=logger,
        coordinator_address=ctrl_config.smpc.coordinator_address,
        jobid=get_smpc_job_id(
            context_id=context_id,
            command_id=command_id,
            operation=op_type,
        ),
        computation_type=op_type,
        clients=smpc_op_clients,
    ) if smpc_op_clients else None

    return True if smpc_op_clients else False


def trigger_smpc_operations(
    logger: Logger,
    context_id: str,
    command_id: int,
    smpc_clients_per_op: Tuple[List[str], List[str], List[str], List[str]],
) -> Tuple[bool, bool, bool, bool]:
    (
        sum_op_smpc_clients,
        min_op_smpc_clients,
        max_op_smpc_clients,
        union_op_smpc_clients,
    ) = smpc_clients_per_op
    sum_op = trigger_smpc_operation(
        logger, context_id, command_id, SMPCRequestType.SUM, sum_op_smpc_clients
    )
    min_op = trigger_smpc_operation(
        logger, context_id, command_id, SMPCRequestType.MIN, min_op_smpc_clients
    )
    max_op = trigger_smpc_operation(
        logger, context_id, command_id, SMPCRequestType.MAX, max_op_smpc_clients
    )
    union_op = trigger_smpc_operation(
        logger, context_id, command_id, SMPCRequestType.UNION, union_op_smpc_clients
    )
    return sum_op, min_op, max_op, union_op


def wait_for_smpc_result_to_be_ready(
    context_id: str,
    command_id: int,
    operation: SMPCRequestType,
):
    jobid = get_smpc_job_id(
        context_id=context_id,
        command_id=command_id,
        operation=operation,
    )

    attempts = 0
    while True:
        sleep(ctrl_config.smpc.get_result_interval)

        response = smpc_cluster.get_smpc_result(
            coordinator_address=ctrl_config.smpc.coordinator_address,
            jobid=jobid,
        )
        try:
            smpc_response = SMPCResponse.parse_raw(response)
        except Exception as exc:
            raise SMPCComputationError(
                f"The SMPC response could not be parsed. \nResponse{response}. \nException: {exc}"
            )

        if smpc_response.status == SMPCResponseStatus.FAILED:
            raise SMPCComputationError(
                f"The SMPC returned a {SMPCResponseStatus.FAILED} status. Body: {response}"
            )
        elif smpc_response.status == SMPCResponseStatus.COMPLETED:
            break

        if attempts > ctrl_config.smpc.get_result_max_retries:
            raise SMPCComputationError(
                f"Max retries for the SMPC exceeded the limit: {ctrl_config.smpc.get_result_max_retries}"
            )
        attempts += 1


def wait_for_smpc_results_to_be_ready(
    context_id: str,
    command_id: int,
    sum_op: bool,
    min_op: bool,
    max_op: bool,
    union_op: bool,
):
    wait_for_smpc_result_to_be_ready(
        context_id, command_id, SMPCRequestType.SUM
    ) if sum_op else None
    wait_for_smpc_result_to_be_ready(
        context_id, command_id, SMPCRequestType.MIN
    ) if min_op else None
    wait_for_smpc_result_to_be_ready(
        context_id, command_id, SMPCRequestType.MAX
    ) if max_op else None
    wait_for_smpc_result_to_be_ready(
        context_id, command_id, SMPCRequestType.UNION
    ) if union_op else None


def get_smpc_results(
    node: GlobalNode,
    context_id: str,
    command_id: int,
    sum_op: bool,
    min_op: bool,
    max_op: bool,
    union_op: bool,
) -> Tuple[TableName, TableName, TableName, TableName]:
    sum_op_result_table = (
        TableName(
            table_name=node.get_smpc_result(
                jobid=get_smpc_job_id(
                    context_id=context_id,
                    command_id=command_id,
                    operation=SMPCRequestType.SUM,
                ),
                command_id=str(command_id),
                command_subid="0",
            )
        )
        if sum_op
        else None
    )
    min_op_result_table = (
        TableName(
            table_name=node.get_smpc_result(
                jobid=get_smpc_job_id(
                    context_id=context_id,
                    command_id=command_id,
                    operation=SMPCRequestType.MIN,
                ),
                command_id=str(command_id),
                command_subid="1",
            )
        )
        if min_op
        else None
    )
    max_op_result_table = (
        TableName(
            table_name=node.get_smpc_result(
                jobid=get_smpc_job_id(
                    context_id=context_id,
                    command_id=command_id,
                    operation=SMPCRequestType.MAX,
                ),
                command_id=str(command_id),
                command_subid="2",
            )
        )
        if max_op
        else None
    )
    union_op_result_table = (
        TableName(
            table_name=node.get_smpc_result(
                jobid=get_smpc_job_id(
                    context_id=context_id,
                    command_id=command_id,
                    operation=SMPCRequestType.UNION,
                ),
                command_id=str(command_id),
                command_subid="3",
            )
        )
        if union_op
        else None
    )

    return (
        sum_op_result_table,
        min_op_result_table,
        max_op_result_table,
        union_op_result_table,
    )
