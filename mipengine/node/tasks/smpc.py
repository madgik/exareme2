import json
from typing import List
from typing import Optional

from celery import shared_task

from mipengine import DType
from mipengine import smpc_cluster_comm_helpers as smpc_cluster
from mipengine.node import config as node_config
from mipengine.node.monetdb_interface.common_actions import create_table_name
from mipengine.node.monetdb_interface.common_actions import get_table_data
from mipengine.node.monetdb_interface.tables import create_table
from mipengine.node.monetdb_interface.tables import insert_data_to_table
from mipengine.node.node_logger import initialise_logger
from mipengine.node_info_DTOs import NodeRole
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import TableType
from mipengine.smpc_cluster_comm_helpers import SMPCComputationError
from mipengine.smpc_DTOs import SMPCRequestType
from mipengine.smpc_DTOs import SMPCResponseWithOutput
from mipengine.table_data_DTOs import ColumnData


@shared_task
@initialise_logger
def validate_smpc_templates_match(
    request_id: str,
    table_name: str,
):
    """
    On a table with multiple SMPC DTO templates, coming from the local nodes,
    it validates that they are all the same without differences.

    Parameters
    ----------
    request_id: The identifier for the logging
    table_name: The table where the templates are located in.

    Returns
    -------
    Nothing, only throws exception if they don't match.
    """

    templates = _get_smpc_values_from_table_data(get_table_data(table_name))
    first_template, *_ = templates
    for template in templates[1:]:
        if template != first_template:
            raise ValueError(
                f"SMPC templates dont match. \n {first_template} \n != \n {template}"
            )


@shared_task
@initialise_logger
def load_data_to_smpc_client(request_id: str, table_name: str, jobid: str) -> str:
    """
    Loads SMPC data into the SMPC client to be used for a computation.

    Parameters
    ----------
    request_id: The identifier for the logging
    table_name: The table where the SMPC op values are located in.
    jobid: An identifier for the SMPC job.

    Returns
    -------
    The id of the client where the data were added
    """
    if node_config.role != NodeRole.LOCALNODE:
        raise PermissionError(
            "load_data_to_smpc_client is allowed only for a LOCALNODE."
        )

    smpc_values, *_ = _get_smpc_values_from_table_data(get_table_data(table_name))

    smpc_cluster.load_data_to_smpc_client(
        node_config.smpc.client_address, jobid, smpc_values
    )

    return node_config.smpc.client_id


@shared_task
@initialise_logger
def get_smpc_result(
    request_id: str,
    jobid: str,
    context_id: str,
    command_id: str,
    command_subid: Optional[str] = "0",
) -> str:
    """
    Fetches the results from an SMPC and writes them into a table.

    Parameters
    ----------
    request_id: The identifier for the logging
    jobid: The identifier for the smpc job.
    context_id: An identifier of the action.
    command_id: An identifier for the command, used for naming the result table.
    command_subid: An identifier for the command, used for naming the result table.
    jobid: The jobid of the SMPC.

    Returns
    -------
    The tablename where the results are in.
    """
    if node_config.role != NodeRole.GLOBALNODE:
        raise PermissionError("get_smpc_result is allowed only for a GLOBALNODE.")

    response = smpc_cluster.get_smpc_result(
        coordinator_address=node_config.smpc.coordinator_address,
        jobid=jobid,
    )

    # We do not need to wait for the result to be ready since the CONTROLLER will do that.
    # The CONTROLLER will trigger this task only when the result is ready.
    try:
        smpc_response = SMPCResponseWithOutput.parse_raw(response)
    except Exception as exc:
        raise SMPCComputationError(
            f"The smpc response could not be parsed into an SMPCResponseWithOutput. "
            f"\nResponse: {response} \nException: {exc}"
        )

    results_table_name = _create_smpc_results_table(
        request_id=request_id,
        context_id=context_id,
        command_id=command_id,
        command_subid=command_subid,
        smpc_op_result_data=smpc_response.computationOutput,
    )

    return results_table_name


def _create_smpc_results_table(
    request_id, context_id, command_id, command_subid, smpc_op_result_data
):
    """
    Create a table with the SMPC specific schema
    and insert the results of the SMPC to it.

    """
    table_name = create_table_name(
        TableType.NORMAL,
        node_config.identifier,
        context_id,
        command_id,
        command_subid,
    )
    table_schema = TableSchema(
        columns=[
            ColumnInfo(
                name="node_id",
                dtype=DType.STR,
            ),
            ColumnInfo(
                name="secure_transfer",
                dtype=DType.JSON,
            ),
        ]
    )
    create_table(table_name, table_schema)

    table_values = [[node_config.identifier, json.dumps(smpc_op_result_data)]]
    insert_data_to_table(table_name, table_values)

    return table_name


def _get_smpc_values_from_table_data(table_data: List[ColumnData]):
    node_id_column, values_column = table_data
    return values_column.data
