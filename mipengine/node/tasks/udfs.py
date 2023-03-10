from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from celery import shared_task

from mipengine.datatypes import DType
from mipengine.node import config as node_config
from mipengine.node.monetdb_interface import udfs
from mipengine.node.monetdb_interface.common_actions import create_table_name
from mipengine.node.monetdb_interface.common_actions import get_table_type
from mipengine.node.monetdb_interface.guard import is_valid_request_id
from mipengine.node.monetdb_interface.guard import output_schema_validator
from mipengine.node.monetdb_interface.guard import sql_injection_guard
from mipengine.node.monetdb_interface.guard import udf_kwargs_validator
from mipengine.node.monetdb_interface.guard import udf_posargs_validator
from mipengine.node.node_logger import initialise_logger
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import NodeLiteralDTO
from mipengine.node_tasks_DTOs import NodeSMPCDTO
from mipengine.node_tasks_DTOs import NodeTableDTO
from mipengine.node_tasks_DTOs import NodeUDFDTO
from mipengine.node_tasks_DTOs import NodeUDFKeyArguments
from mipengine.node_tasks_DTOs import NodeUDFPosArguments
from mipengine.node_tasks_DTOs import NodeUDFResults
from mipengine.node_tasks_DTOs import SMPCTablesInfo
from mipengine.node_tasks_DTOs import TableInfo
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import TableType
from mipengine.smpc_cluster_comm_helpers import validate_smpc_usage
from mipengine.udfgen import FlowUdfArg
from mipengine.udfgen import get_udfgenerator
from mipengine.udfgen import udf
from mipengine.udfgen.udfgen_DTOs import UDFGenResult
from mipengine.udfgen.udfgen_DTOs import UDFGenSMPCResult
from mipengine.udfgen.udfgen_DTOs import UDFGenTableResult


@shared_task
@initialise_logger
def get_udf(request_id: str, func_name: str) -> str:
    return str(udf.registry[func_name])


@shared_task
@initialise_logger
def run_udf(
    request_id: str,
    command_id: str,
    context_id: str,
    func_name: str,
    positional_args_json: str,
    keyword_args_json: str,
    use_smpc: bool = False,
    output_schema: Optional[str] = None,
) -> str:
    """
    Creates the UDF, if provided, and adds it in the database.
    Then it runs the select statement with the input provided.

    Parameters
    ----------
        request_id : str
            The identifier for the logging
        command_id: str
            The command identifier, common among all nodes for this action.
        context_id: str
            The experiment identifier, common among all experiment related actions.
        func_name: str
            Name of function from which to generate UDF.
        positional_args_json: str(NodeUDFPosArguments)
            Positional arguments of the udf call.
        keyword_args_json: str(NodeUDFKeyArguments)
            Keyword arguments of the udf call.
        use_smpc: bool
            Should SMPC be used?
        output_schema: Optional[str(TableSchema)]
            Schema of main UDF output when deferred mechanism is used.
    Returns
    -------
        str(NodeUDFResults)
            The results, with the tablenames, that the execution created.
    """
    validate_smpc_usage(use_smpc, node_config.smpc.enabled, node_config.smpc.optional)

    positional_args = NodeUDFPosArguments.parse_raw(positional_args_json)
    keyword_args = NodeUDFKeyArguments.parse_raw(keyword_args_json)

    if output_schema is not None:
        output_schema = _convert_output_schema(output_schema)

    udf_definitions, udf_exec_stmt, udf_results = _generate_udf_statements(
        request_id=request_id,
        command_id=command_id,
        context_id=context_id,
        func_name=func_name,
        positional_args=positional_args,
        keyword_args=keyword_args,
        use_smpc=use_smpc,
        output_schema=output_schema,
    )

    udfs.run_udf(udf_definitions, udf_exec_stmt)

    return udf_results.json()


def _convert_output_schema(output_schema: str) -> List[Tuple[str, DType]]:
    table_schema = TableSchema.parse_raw(output_schema)
    return [(col.name, col.dtype) for col in table_schema.columns]


@shared_task
@initialise_logger
def get_run_udf_query(
    command_id: str,
    request_id: str,
    context_id: str,
    func_name: str,
    positional_args_json: str,
    keyword_args_json: str,
    use_smpc: bool = False,
) -> List[str]:
    """
    Fetches the sql statements that represent the execution of the udf.

    Parameters
    ----------
        command_id: str
            The command identifier, common among all nodes for this action.
        request_id : str
            The identifier for the logging
        context_id: str
            The experiment identifier, common among all experiment related actions.
        func_name: str
            Name of function from which to generate UDF.
        positional_args_json: str(UDFPosArguments)
            Positional arguments of the udf call.
        keyword_args_json: str(UDFKeyArguments)
            Keyword arguments of the udf call.
        use_smpc: bool
            Should SMPC be used?
    Returns
    -------
        List[str]
            A list of the statements that would be executed in the DB.

    """
    # TODO why should we validate here?
    validate_smpc_usage(use_smpc, node_config.smpc.enabled, node_config.smpc.optional)

    positional_args = NodeUDFPosArguments.parse_raw(positional_args_json)
    keyword_args = NodeUDFKeyArguments.parse_raw(keyword_args_json)

    udf_definitions, udf_exec_stmt, _ = _generate_udf_statements(
        request_id=request_id,
        command_id=command_id,
        context_id=context_id,
        func_name=func_name,
        positional_args=positional_args,
        keyword_args=keyword_args,
        use_smpc=use_smpc,
    )

    return udf_definitions + [udf_exec_stmt]


def _create_udf_name(func_name: str, command_id: str, context_id: str) -> str:
    """
    Creates a udf name with the format <func_name>_<commandId>_<contextId>
    """
    # TODO Monetdb UDF name cannot be larger than 63 character
    return f"{func_name}_{command_id}_{context_id}"


def _convert_nodeudf_to_flow_args(
    positional_args: NodeUDFPosArguments,
    keyword_args: NodeUDFKeyArguments,
) -> Tuple[List[FlowUdfArg], Dict[str, FlowUdfArg]]:
    """
    Converts UDF arguments DTOs in format understood by UDF generator

    The input arguments are received from the controller and contain the value
    of the argument (literal) or information about the location of the input
    (tablename). This function creates new objects with added information
    necessary for the UDF generator. These objects are named FlowUdfArgs
    because, in the context of the UDF generator, they come from the algorithm
    flow.

    Parameters
    ----------
    positional_args : NodeUDFPosArguments
        The pos arguments received from the controller.
    keyword_args : NodeUDFKeyArguments
        The kw arguments received from the controller.

    Returns
    -------
    List[FlowUdfArg]
        Args for the UDF generator.
    Dict[str, FlowUdfArg]
        Kwargs for the UDF generator.
    """

    def convert(arg: NodeUDFDTO) -> FlowUdfArg:
        if isinstance(arg, NodeTableDTO):
            _validate_tableinfo_type_matches_actual_tabletype(arg.value)
            return arg.value
        elif isinstance(arg, NodeSMPCDTO):
            _validate_smpctablesinfo_type_matches_actual_tablestype(arg.value)
            return arg.value
        elif isinstance(arg, NodeLiteralDTO):
            return arg.value
        raise ValueError(f"A UDF argument needs to be an instance of {NodeUDFDTO}'.")

    flowargs = [convert(arg) for arg in positional_args.args]

    flowkwargs = {key: convert(arg) for key, arg in keyword_args.args.items()}

    return flowargs, flowkwargs


def _validate_tableinfo_type_matches_actual_tabletype(table_info: TableInfo):
    if table_info.type_ != get_table_type(table_info.name):
        msg = f"Table: '{table_info.name}' is not of type: '{table_info.type_}'."
        raise ValueError(msg)


def _validate_smpctablesinfo_type_matches_actual_tablestype(
    tables_info: SMPCTablesInfo,
):
    _validate_tableinfo_type_matches_actual_tabletype(tables_info.template)
    if tables_info.sum_op:
        _validate_tableinfo_type_matches_actual_tabletype(tables_info.sum_op)
    if tables_info.min_op:
        _validate_tableinfo_type_matches_actual_tabletype(tables_info.min_op)
    if tables_info.max_op:
        _validate_tableinfo_type_matches_actual_tabletype(tables_info.max_op)


@sql_injection_guard(
    request_id=is_valid_request_id,
    command_id=str.isalnum,
    context_id=str.isalnum,
    func_name=str.isidentifier,
    positional_args=udf_posargs_validator,
    keyword_args=udf_kwargs_validator,
    use_smpc=None,
    output_schema=output_schema_validator,
)
def _generate_udf_statements(
    request_id: str,
    command_id: str,
    context_id: str,
    func_name: str,
    positional_args: NodeUDFPosArguments,
    keyword_args: NodeUDFKeyArguments,
    use_smpc: bool,
    output_schema,
) -> Tuple[List[str], str, NodeUDFResults]:
    # Data needed for UDF generation
    # ------------------------------
    flowargs, flowkwargs = _convert_nodeudf_to_flow_args(positional_args, keyword_args)
    udf_name = _create_udf_name(func_name, command_id, context_id)

    # node_id is needed for table name creation
    node_id = node_config.identifier

    # min_row_count is necessary when an algorithm needs it in the UDF
    min_row_count = node_config.privacy.minimum_row_count

    udfgen = get_udfgenerator(
        udfregistry=udf.registry,
        func_name=func_name,
        flowargs=flowargs,
        flowkwargs=flowkwargs,
        smpc_used=use_smpc,
        request_id=request_id,
        output_schema=output_schema,
        min_row_count=min_row_count,
    )
    # outputnum is the number of UDF outputs, we need it to create an
    # equal number of output names before calling the UDF generator
    outputnum = udfgen.num_outputs

    # A UDF may produce more than one table results, so we create a
    # list of one or more output table names
    output_names = _make_output_table_names(outputnum, node_id, context_id, command_id)

    # UDF generation
    # --------------
    udf_definition = udfgen.get_definition(udf_name, output_names)
    udf_exec_stmt = udfgen.get_exec_stmt(udf_name, output_names)
    udf_results = udfgen.get_results(output_names)

    # Create list of udf statements
    udf_definitions = [res.create_query for res in udf_results]
    udf_definitions.append(udf_definition)

    # Convert results
    results = [_convert_result(res) for res in udf_results]
    results_dto = NodeUDFResults(results=results)

    return udf_definitions, udf_exec_stmt, results_dto


def _make_output_table_names(
    outputlen: int, node_id: str, context_id: str, command_id: str
) -> List[str]:
    return [
        create_table_name(
            table_type=TableType.NORMAL,
            node_id=node_id,
            context_id=context_id,
            command_id=command_id,
            result_id=str(id),
        )
        for id in range(outputlen)
    ]


def _convert_result(result: UDFGenResult) -> NodeUDFDTO:
    if isinstance(result, UDFGenTableResult):
        return _convert_table_result(result)
    elif isinstance(result, UDFGenSMPCResult):
        return _convert_smpc_result(result)
    raise TypeError(f"Unknown result type {result.__class__}")


def _convert_table_result(result: UDFGenTableResult) -> NodeTableDTO:
    table_info = TableInfo(
        name=result.table_name,
        schema_=_convert_result_schema(result.table_schema),
        type_=TableType.NORMAL,
    )
    return NodeTableDTO(value=table_info)


def _convert_smpc_result(result: UDFGenSMPCResult) -> NodeSMPCDTO:
    table_infos = {}

    table_infos["template"] = TableInfo(
        name=result.template.table_name,
        schema_=_convert_result_schema(result.template.table_schema),
        type_=TableType.NORMAL,
    )

    if result.sum_op_values:
        table_infos["sum_op"] = TableInfo(
            name=result.sum_op_values.table_name,
            schema_=_convert_result_schema(result.sum_op_values.table_schema),
            type_=TableType.NORMAL,
        )

    if result.min_op_values:
        table_infos["min_op"] = TableInfo(
            name=result.min_op_values.table_name,
            schema_=_convert_result_schema(result.min_op_values.table_schema),
            type_=TableType.NORMAL,
        )

    if result.max_op_values:
        table_infos["max_op"] = TableInfo(
            name=result.max_op_values.table_name,
            schema_=_convert_result_schema(result.max_op_values.table_schema),
            type_=TableType.NORMAL,
        )

    return NodeSMPCDTO(value=SMPCTablesInfo(**table_infos))


def _convert_result_schema(schema: List[Tuple[str, DType]]) -> TableSchema:
    columns = [ColumnInfo(name=name, dtype=dtype) for name, dtype in schema]
    return TableSchema(columns=columns)
