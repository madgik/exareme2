from typing import List
from typing import Tuple

from mipengine.controller import config as ctrl_config
from mipengine.controller.algorithm_executor_node_data_objects import GlobalNodeTable
from mipengine.controller.algorithm_executor_node_data_objects import (
    LocalNodesSMPCTables,
)
from mipengine.controller.algorithm_executor_node_data_objects import LocalNodesTable
from mipengine.controller.algorithm_executor_node_data_objects import NodeTable
from mipengine.controller.algorithm_executor_nodes import GlobalNode
from mipengine.smpc_DTOs import SMPCRequestType
from mipengine.smpc_cluster_comm_helpers import trigger_smpc_computation


def get_smpc_job_id(
    context_id: str, command_id: int, operation: SMPCRequestType
) -> str:
    return context_id + "_" + str(command_id) + "_" + str(operation)


def load_operation_data_to_smpc_clients(
    command_id: int, local_nodes_table: LocalNodesTable, op_type: SMPCRequestType
) -> List[int]:
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
) -> Tuple[List[int], List[int], List[int], List[int]]:
    sum_op_smpc_clients = load_operation_data_to_smpc_clients(
        command_id, smpc_tables.sum_op, SMPCRequestType.SUM
    )
    min_op_smpc_clients = load_operation_data_to_smpc_clients(
        command_id, smpc_tables.min_op, SMPCRequestType.MIN
    )
    max_op_smpc_clients = load_operation_data_to_smpc_clients(
        command_id, smpc_tables.max_op, SMPCRequestType.MAX
    )
    union_op_smpc_clients = load_operation_data_to_smpc_clients(
        command_id, smpc_tables.union_op, SMPCRequestType.UNION
    )
    return (
        sum_op_smpc_clients,
        min_op_smpc_clients,
        max_op_smpc_clients,
        union_op_smpc_clients,
    )


def trigger_smpc_operation_computation(
    context_id: str,
    command_id: int,
    op_type: SMPCRequestType,
    smpc_op_clients: List[int],
) -> bool:
    trigger_smpc_computation(
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


def trigger_smpc_computations(
    context_id: str,
    command_id: int,
    smpc_clients_per_op: Tuple[List[int], List[int], List[int], List[int]],
) -> Tuple[bool, bool, bool, bool]:
    (
        sum_op_smpc_clients,
        min_op_smpc_clients,
        max_op_smpc_clients,
        union_op_smpc_clients,
    ) = smpc_clients_per_op
    sum_op = trigger_smpc_operation_computation(
        context_id, command_id, SMPCRequestType.SUM, sum_op_smpc_clients
    )
    min_op = trigger_smpc_operation_computation(
        context_id, command_id, SMPCRequestType.MIN, min_op_smpc_clients
    )
    max_op = trigger_smpc_operation_computation(
        context_id, command_id, SMPCRequestType.MAX, max_op_smpc_clients
    )
    union_op = trigger_smpc_operation_computation(
        context_id, command_id, SMPCRequestType.UNION, union_op_smpc_clients
    )
    return sum_op, min_op, max_op, union_op


def get_smpc_results(
    node: GlobalNode,
    context_id: str,
    command_id: int,
    sum_op: bool,
    min_op: bool,
    max_op: bool,
    union_op: bool,
) -> Tuple[GlobalNodeTable, GlobalNodeTable, GlobalNodeTable, GlobalNodeTable]:
    sum_op_result_table = (
        node.get_smpc_result(
            command_id=command_id,
            jobid=get_smpc_job_id(
                context_id=context_id,
                command_id=command_id,
                operation=SMPCRequestType.SUM,
            ),
        )
        if sum_op
        else None
    )
    min_op_result_table = (
        node.get_smpc_result(
            command_id=command_id,
            jobid=get_smpc_job_id(
                context_id=context_id,
                command_id=command_id,
                operation=SMPCRequestType.MIN,
            ),
        )
        if min_op
        else None
    )
    max_op_result_table = (
        node.get_smpc_result(
            command_id=command_id,
            jobid=get_smpc_job_id(
                context_id=context_id,
                command_id=command_id,
                operation=SMPCRequestType.MAX,
            ),
        )
        if max_op
        else None
    )
    union_op_result_table = (
        node.get_smpc_result(
            command_id=command_id,
            jobid=get_smpc_job_id(
                context_id=context_id,
                command_id=command_id,
                operation=SMPCRequestType.UNION,
            ),
        )
        if union_op
        else None
    )

    result = (
        GlobalNodeTable(node=node, table=NodeTable(table_name=sum_op_result_table))
        if sum_op_result_table
        else None,
        GlobalNodeTable(node=node, table=NodeTable(table_name=min_op_result_table))
        if min_op_result_table
        else None,
        GlobalNodeTable(node=node, table=NodeTable(table_name=max_op_result_table))
        if max_op_result_table
        else None,
        GlobalNodeTable(node=node, table=NodeTable(table_name=union_op_result_table))
        if union_op_result_table
        else None,
    )

    return result
