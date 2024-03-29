from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from exareme2.algorithms.exareme2.udfgen import FlowUdfArg
from exareme2.algorithms.exareme2.udfgen import get_udfgenerator
from exareme2.algorithms.exareme2.udfgen import udf
from exareme2.algorithms.exareme2.udfgen.udfgen_DTOs import UDFGenResult
from exareme2.algorithms.exareme2.udfgen.udfgen_DTOs import UDFGenSMPCResult
from exareme2.algorithms.exareme2.udfgen.udfgen_DTOs import UDFGenTableResult
from exareme2.datatypes import DType
from exareme2.smpc_cluster_communication import validate_smpc_usage
from exareme2.worker import config as worker_config
from exareme2.worker.exareme2.monetdb.guard import is_valid_request_id
from exareme2.worker.exareme2.monetdb.guard import output_schema_validator
from exareme2.worker.exareme2.monetdb.guard import sql_injection_guard
from exareme2.worker.exareme2.monetdb.guard import udf_kwargs_validator
from exareme2.worker.exareme2.monetdb.guard import udf_posargs_validator
from exareme2.worker.exareme2.tables.tables_db import create_table_name
from exareme2.worker.exareme2.tables.tables_db import get_table_type
from exareme2.worker.exareme2.udfs import udfs_db
from exareme2.worker.utils.logger import initialise_logger
from exareme2.worker_communication import NodeLiteralDTO
from exareme2.worker_communication import NodeSMPCDTO
from exareme2.worker_communication import NodeTableDTO
from exareme2.worker_communication import NodeUDFDTO
from exareme2.worker_communication import NodeUDFKeyArguments
from exareme2.worker_communication import NodeUDFPosArguments
from exareme2.worker_communication import NodeUDFResults
from exareme2.worker_communication import SMPCTablesInfo
from exareme2.worker_communication import TableInfo
from exareme2.worker_communication import TableSchema
from exareme2.worker_communication import TableType


@initialise_logger
def run_udf(
    request_id: str,
    command_id: str,
    context_id: str,
    func_name: str,
    positional_args: NodeUDFPosArguments,
    keyword_args: NodeUDFKeyArguments,
    use_smpc: bool = False,
    output_schema: Optional[str] = None,
) -> NodeUDFResults:
    """
    Creates the UDF, if provided, and adds it in the database.
    Then it runs the select statement with the input provided.

    Parameters
    ----------
        request_id : str
            The identifier for the logging
        command_id: str
            The command identifier, common among all workers for this action.
        context_id: str
            The experiment identifier, common among all experiment related actions.
        func_name: str
            Name of function from which to generate UDF.
        positional_args: NodeUDFPosArguments
            Positional arguments of the udf call.
        keyword_args: NodeUDFKeyArguments
            Keyword arguments of the udf call.
        use_smpc: bool
            Should SMPC be used?
        output_schema: Optional[str(TableSchema)]
            Schema of main UDF output when deferred mechanism is used.
    Returns
    -------
        NodeUDFResults
            The results, with the tablenames, that the execution created.
    """
    validate_smpc_usage(
        use_smpc, worker_config.smpc.enabled, worker_config.smpc.optional
    )

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

    udfs_db.run_udf(udf_definitions, udf_exec_stmt)

    return udf_results


def _convert_output_schema(output_schema: str) -> List[Tuple[str, DType]]:
    table_schema = TableSchema.parse_raw(output_schema)
    return table_schema.to_list()


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

    # worker_id is needed for table name creation
    worker_id = worker_config.identifier

    # min_row_count is necessary when an algorithm needs it in the UDF
    min_row_count = worker_config.privacy.minimum_row_count

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
    output_names = _make_output_table_names(
        outputnum, worker_id, context_id, command_id
    )

    # UDF generation
    udf_definition = udfgen.get_definition(udf_name, output_names)
    udf_exec_stmt = udfgen.get_exec_stmt(udf_name, output_names)
    udf_results = udfgen.get_results(output_names)

    # Create list of udf definitions
    table_creation_queries = _get_udf_table_creation_queries(udf_results)
    public_username = worker_config.monetdb.public_username
    table_sharing_queries = _get_udf_table_sharing_queries(udf_results, public_username)
    udf_definitions = [*table_creation_queries, *table_sharing_queries, udf_definition]

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


def _get_udf_table_creation_queries(udf_results: List[UDFGenResult]) -> List[str]:
    queries = []
    for result in udf_results:
        if isinstance(result, UDFGenTableResult):
            queries.append(result.create_query)
        elif isinstance(result, UDFGenSMPCResult):
            queries.extend(_get_udf_smpc_table_creation_queries(result))
        else:
            raise NotImplementedError
    return queries


def _get_udf_smpc_table_creation_queries(result: UDFGenSMPCResult) -> List[str]:
    assert isinstance(result, UDFGenSMPCResult)
    queries = [result.template.create_query]
    if result.sum_op_values is not None:
        queries.append(result.sum_op_values.create_query)
    if result.min_op_values is not None:
        queries.append(result.min_op_values.create_query)
    if result.max_op_values is not None:
        queries.append(result.max_op_values.create_query)
    return queries


def _get_table_share_query(tablename: str, public_username: str) -> str:
    return f"GRANT SELECT ON TABLE {tablename} TO {public_username}"


def _get_udf_table_sharing_queries(
    udf_results: List[UDFGenResult], public_username
) -> List[str]:
    """
    Tables should be shared (accessible through the "public" user) in the following cases:
    1) The result is of UDFGenTableResult type and the share property is True,
    2) The result is of UDFGenSMPCResult type, so the template should be shared, the rest of the tables
        will pass through the SMPC.
    """
    queries = []
    for result in udf_results:
        if isinstance(result, UDFGenTableResult):
            queries.append(
                _get_table_share_query(result.table_name, public_username)
            ) if result.share else None
        elif isinstance(result, UDFGenSMPCResult):
            queries.append(
                _get_table_share_query(result.template.table_name, public_username)
            )
        else:
            raise NotImplementedError
    return queries


def _convert_result(result: UDFGenResult) -> NodeUDFDTO:
    if isinstance(result, UDFGenTableResult):
        return _convert_table_result(result)
    elif isinstance(result, UDFGenSMPCResult):
        return _convert_smpc_result(result)
    raise TypeError(f"Unknown result type {result.__class__}")


def _convert_table_result(result: UDFGenTableResult) -> NodeTableDTO:
    table_info = TableInfo(
        name=result.table_name,
        schema_=TableSchema.from_list(result.table_schema),
        type_=TableType.NORMAL,
    )
    return NodeTableDTO(value=table_info)


def _convert_smpc_result(result: UDFGenSMPCResult) -> NodeSMPCDTO:
    table_infos = {}

    table_infos["template"] = TableInfo(
        name=result.template.table_name,
        schema_=TableSchema.from_list(result.template.table_schema),
        type_=TableType.NORMAL,
    )

    if result.sum_op_values:
        table_infos["sum_op"] = TableInfo(
            name=result.sum_op_values.table_name,
            schema_=TableSchema.from_list(result.sum_op_values.table_schema),
            type_=TableType.NORMAL,
        )

    if result.min_op_values:
        table_infos["min_op"] = TableInfo(
            name=result.min_op_values.table_name,
            schema_=TableSchema.from_list(result.min_op_values.table_schema),
            type_=TableType.NORMAL,
        )

    if result.max_op_values:
        table_infos["max_op"] = TableInfo(
            name=result.max_op_values.table_name,
            schema_=TableSchema.from_list(result.max_op_values.table_schema),
            type_=TableType.NORMAL,
        )

    return NodeSMPCDTO(value=SMPCTablesInfo(**table_infos))
