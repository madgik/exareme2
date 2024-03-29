import json
from typing import List
from typing import Optional

from exareme2 import DType
from exareme2 import smpc_cluster_communication as smpc_cluster
from exareme2.node import config as node_config
from exareme2.node.logger import initialise_logger
from exareme2.node.monetdb.tables import create_table
from exareme2.node.monetdb.tables import create_table_name
from exareme2.node.monetdb.tables import get_table_data
from exareme2.node.monetdb.tables import insert_data_to_table
from exareme2.node_communication import ColumnData
from exareme2.node_communication import ColumnInfo
from exareme2.node_communication import NodeRole
from exareme2.node_communication import TableInfo
from exareme2.node_communication import TableSchema
from exareme2.node_communication import TableType
from exareme2.smpc_cluster_communication import SMPCComputationError
from exareme2.smpc_cluster_communication import SMPCResponseWithOutput
from exareme2.smpc_cluster_communication import SMPCUsageError


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

    templates = _get_smpc_values_from_table_data(get_table_data(table_name, False))
    first_template, *_ = templates
    for template in templates[1:]:
        if template != first_template:
            raise ValueError(
                f"SMPC templates dont match. \n {first_template} \n != \n {template}"
            )


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

    smpc_values, *_ = _get_smpc_values_from_table_data(
        get_table_data(table_name, False)
    )

    smpc_cluster.load_data_to_smpc_client(
        node_config.smpc.client_address, jobid, smpc_values
    )

    return node_config.smpc.client_id


@initialise_logger
def get_smpc_result(
    request_id: str,
    jobid: str,
    context_id: str,
    command_id: str,
    command_subid: Optional[str] = "0",
) -> TableInfo:
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

    results_table_name, results_table_schema = _create_smpc_results_table(
        request_id=request_id,
        context_id=context_id,
        command_id=command_id,
        command_subid=command_subid,
        smpc_op_result_data=smpc_response.computationOutput,
    )

    return TableInfo(
        name=results_table_name,
        schema_=results_table_schema,
        type_=TableType.NORMAL,
    )


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
                name="secure_transfer",
                dtype=DType.JSON,
            ),
        ]
    )
    create_table(table_name, table_schema)

    table_values = [[json.dumps(smpc_op_result_data)]]
    insert_data_to_table(table_name, table_values)

    return table_name, table_schema


def _get_smpc_values_from_table_data(table_data: List[ColumnData]):
    values_column, *_ = table_data

    if not values_column.data:
        raise SMPCUsageError("A node doesn't have data to contribute to the SMPC.")

    return values_column.data
