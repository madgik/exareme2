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
from mipengine.node_tasks_DTOs import _NodeUDFDTOType
from mipengine.smpc_cluster_comm_helpers import validate_smpc_usage
from mipengine.udfgen import generate_udf_queries
from mipengine.udfgen.udfgen_DTOs import UDFGenExecutionQueries
from mipengine.udfgen.udfgen_DTOs import UDFGenResult
from mipengine.udfgen.udfgen_DTOs import UDFGenSMPCResult
from mipengine.udfgen.udfgen_DTOs import UDFGenTableResult
from mipengine.udfgen.udfgenerator import UDFGenArgument
from mipengine.udfgen.udfgenerator import udf as udf_registry


@shared_task
@initialise_logger
def get_udf(request_id: str, func_name: str) -> str:
    return str(udf_registry.registry[func_name])


# TODO https://team-1617704806227.atlassian.net/browse/MIP-473
# @shared_task(
#     soft_time_limit=node_config.celery.run_udf_soft_time_limit,
#     time_limit=node_config.celery.run_udf_time_limit,
# )
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

    if output_schema:
        output_schema = _convert_tableschema2udfgen_iotype(
            TableSchema.parse_raw(output_schema)
        )

    udf_statements, udf_results = _generate_udf_statements(
        request_id=request_id,
        command_id=command_id,
        context_id=context_id,
        func_name=func_name,
        positional_args=positional_args,
        keyword_args=keyword_args,
        use_smpc=use_smpc,
        output_schema=output_schema,
    )

    udfs.run_udf(udf_statements)

    return udf_results.json()


def _convert_tableschema2udfgen_iotype(
    output_schema: TableSchema,
) -> List[Tuple[str, DType]]:
    return [(col.name, col.dtype) for col in output_schema.columns]


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
    validate_smpc_usage(use_smpc, node_config.smpc.enabled, node_config.smpc.optional)

    positional_args = NodeUDFPosArguments.parse_raw(positional_args_json)
    keyword_args = NodeUDFKeyArguments.parse_raw(keyword_args_json)

    udf_statements, _ = _generate_udf_statements(
        request_id=request_id,
        command_id=command_id,
        context_id=context_id,
        func_name=func_name,
        positional_args=positional_args,
        keyword_args=keyword_args,
        use_smpc=use_smpc,
    )

    return udf_statements


def _create_udf_name(func_name: str, command_id: str, context_id: str) -> str:
    """
    Creates a udf name with the format <func_name>_<commandId>_<contextId>
    """
    # TODO Monetdb UDF name cannot be larger than 63 character
    return f"{func_name}_{command_id}_{context_id}"


def _convert_nodeudf2udfgen_args(
    positional_args: NodeUDFPosArguments,
    keyword_args: NodeUDFKeyArguments,
) -> Tuple[List[UDFGenArgument], Dict[str, UDFGenArgument]]:
    """
    The input arguments are received from the controller and contain
    the value of the argument (literal) or information
    about the location of the input (tablename).

    This method adds information on these arguments, that will be
    sent to the udfgenerator.

    The udfgen args can be of number, TableInfo or SMPCUDFInput type.

    Parameters
    ----------
    positional_args The pos arguments received from the controller.
    keyword_args The kw arguments received from the controller.

    Returns
    -------
    The same arguments (pos/kw) in a udfgen argument structure.
    """
    generator_pos_args = [
        _convert_nodeudf2udfgen_arg(pos_arg) for pos_arg in positional_args.args
    ]

    generator_kw_args = {
        key: _convert_nodeudf2udfgen_arg(argument)
        for key, argument in keyword_args.args.items()
    }

    return generator_pos_args, generator_kw_args


def _convert_nodeudf2udfgen_arg(udf_argument: NodeUDFDTO) -> UDFGenArgument:
    if isinstance(udf_argument, NodeTableDTO):
        validate_tableinfo_type_matches_actual_tabletype(udf_argument.value)
        return udf_argument.value
    elif isinstance(udf_argument, NodeSMPCDTO):
        validate_smpctablesinfo_type_matches_actual_tablestype(udf_argument.value)
        return udf_argument.value
    elif isinstance(udf_argument, NodeLiteralDTO):
        return udf_argument.value
    else:
        argument_kinds = ",".join([str(k) for k in _NodeUDFDTOType])
        raise ValueError(
            f"A udf argument can have one of the following types {argument_kinds}'."
        )


def validate_tableinfo_type_matches_actual_tabletype(table_info: TableInfo):
    if table_info.type_ != get_table_type(table_info.name):
        raise ValueError(
            f"Table: '{table_info.name}' is not of type: '{table_info.type_}'."
        )


def validate_smpctablesinfo_type_matches_actual_tablestype(tables_info: SMPCTablesInfo):
    validate_tableinfo_type_matches_actual_tabletype(tables_info.template)
    validate_tableinfo_type_matches_actual_tabletype(
        tables_info.sum_op
    ) if tables_info.sum_op else None
    validate_tableinfo_type_matches_actual_tabletype(
        tables_info.min_op
    ) if tables_info.min_op else None
    validate_tableinfo_type_matches_actual_tabletype(
        tables_info.max_op
    ) if tables_info.max_op else None


def _convert_table_result_to_udf_statements(
    result: UDFGenTableResult, templates_mapping: dict
) -> List[str]:
    return [
        result.drop_query.substitute(**templates_mapping),
        result.create_query.substitute(**templates_mapping),
    ]


def _get_all_table_results_from_smpc_result(
    smpc_result: UDFGenSMPCResult,
) -> List[UDFGenTableResult]:
    table_results = [smpc_result.template]
    table_results.append(
        smpc_result.sum_op_values
    ) if smpc_result.sum_op_values else None
    table_results.append(
        smpc_result.min_op_values
    ) if smpc_result.min_op_values else None
    table_results.append(
        smpc_result.max_op_values
    ) if smpc_result.max_op_values else None
    return table_results


def _convert_smpc_result_to_udf_statements(
    result: UDFGenSMPCResult, templates_mapping: dict
) -> List[str]:
    table_results = _get_all_table_results_from_smpc_result(result)
    udf_statements = []
    for result in table_results:
        udf_statements.extend(
            _convert_table_result_to_udf_statements(result, templates_mapping)
        )
    return udf_statements


def _create_udf_statements(
    udf_execution_queries: UDFGenExecutionQueries,
    templates_mapping: dict,
) -> List[str]:
    udf_statements = []
    for result in udf_execution_queries.udf_results:
        if isinstance(result, UDFGenTableResult):
            udf_statements.extend(
                _convert_table_result_to_udf_statements(result, templates_mapping)
            )
        elif isinstance(result, UDFGenSMPCResult):
            udf_statements.extend(
                _convert_smpc_result_to_udf_statements(result, templates_mapping)
            )
        else:
            raise NotImplementedError

    if udf_execution_queries.udf_definition_query:
        udf_statements.append(
            udf_execution_queries.udf_definition_query.substitute(**templates_mapping)
        )
    udf_statements.append(
        udf_execution_queries.udf_select_query.substitute(**templates_mapping)
    )

    return udf_statements


def _convert_udfgen2table_info_result_and_mapping(
    udfgen_result: UDFGenTableResult,
    context_id: str,
    command_id: str,
    command_subid: int,
) -> Tuple[TableInfo, Dict[str, str]]:
    table_name_ = create_table_name(
        table_type=TableType.NORMAL,
        node_id=node_config.identifier,
        context_id=context_id,
        command_id=command_id,
        command_subid=str(command_subid),
    )
    table_name_tmpl_mapping = {udfgen_result.tablename_placeholder: table_name_}
    return (
        TableInfo(
            name=table_name_,
            schema_=_convert_udfgen_iotype2table_schema(udfgen_result.table_schema),
            type_=TableType.NORMAL,
        ),
        table_name_tmpl_mapping,
    )


def _convert_udfgen_iotype2table_schema(iotype: List[Tuple[str, DType]]) -> TableSchema:
    return TableSchema(
        columns=[ColumnInfo(name=name, dtype=dtype) for name, dtype in iotype]
    )


def _convert_udfgen2smpc_tables_info_result_and_mapping(
    udfgen_result: UDFGenSMPCResult,
    context_id: str,
    command_id: str,
    command_subid: int,
) -> Tuple[SMPCTablesInfo, Dict[str, str]]:
    (
        template_udf_result,
        table_names_tmpl_mapping,
    ) = _convert_udfgen2table_info_result_and_mapping(
        udfgen_result.template, context_id, command_id, command_subid
    )

    if udfgen_result.sum_op_values:
        (sum_op_udf_result, mapping,) = _convert_udfgen2table_info_result_and_mapping(
            udfgen_result.sum_op_values, context_id, command_id, command_subid + 1
        )
        table_names_tmpl_mapping.update(mapping)
    else:
        sum_op_udf_result = None

    if udfgen_result.min_op_values:
        (min_op_udf_result, mapping,) = _convert_udfgen2table_info_result_and_mapping(
            udfgen_result.min_op_values, context_id, command_id, command_subid + 2
        )
        table_names_tmpl_mapping.update(mapping)
    else:
        min_op_udf_result = None

    if udfgen_result.max_op_values:
        (max_op_udf_result, mapping,) = _convert_udfgen2table_info_result_and_mapping(
            udfgen_result.max_op_values, context_id, command_id, command_subid + 3
        )
        table_names_tmpl_mapping.update(mapping)
    else:
        max_op_udf_result = None

    result = SMPCTablesInfo(
        template=template_udf_result,
        sum_op=sum_op_udf_result,
        min_op=min_op_udf_result,
        max_op=max_op_udf_result,
    )
    return result, table_names_tmpl_mapping


def _convert_udfgen2nodeudf_result_and_mapping(
    udfgen_result: UDFGenResult,
    context_id: str,
    command_id: str,
    command_subid: int,
) -> Tuple[NodeUDFDTO, Dict[str, str]]:
    if isinstance(udfgen_result, UDFGenTableResult):
        table_info, mapping = _convert_udfgen2table_info_result_and_mapping(
            udfgen_result, context_id, command_id, command_subid
        )
        return NodeTableDTO(value=table_info), mapping
    elif isinstance(udfgen_result, UDFGenSMPCResult):
        tables_info, mapping = _convert_udfgen2smpc_tables_info_result_and_mapping(
            udfgen_result, context_id, command_id, command_subid
        )
        return NodeSMPCDTO(value=tables_info), mapping
    else:
        raise NotImplementedError


def convert_udfgen2nodeudf_results_and_mapping(
    udf_queries: UDFGenExecutionQueries,
    context_id: str,
    command_id: str,
) -> Tuple[NodeUDFResults, Dict[str, str]]:
    """
    Iterates through all the udf generator results, in order to create
    a table for each one.

    UDFResults are returned together with a mapping of
    template -> tablename, so it can be used in the udf's declaration to
    replace the templates with the actual table names.

    Returns
    -------
    a UDFResults object containing all the results
    a dictionary of template (placeholder) tablename to the actual table name.
    """
    results = []
    table_names_tmpl_mapping = {}
    command_subid = 0
    for udf_result in udf_queries.udf_results:
        table, mapping = _convert_udfgen2nodeudf_result_and_mapping(
            udf_result,
            context_id,
            command_id,
            command_subid,
        )
        table_names_tmpl_mapping.update(mapping)
        results.append(table)

        # Needs to be incremented by 10 because a udf_result could
        # contain more than one tables. (SMPC for example)
        command_subid += 10

    udf_results = NodeUDFResults(results=results)
    return udf_results, table_names_tmpl_mapping


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
) -> Tuple[List[str], NodeUDFResults]:
    udf_name = _create_udf_name(func_name, command_id, context_id)

    gen_pos_args, gen_kw_args = _convert_nodeudf2udfgen_args(
        positional_args, keyword_args
    )

    udf_execution_queries = generate_udf_queries(
        func_name=func_name,
        positional_args=gen_pos_args,
        keyword_args=gen_kw_args,
        smpc_used=use_smpc,
        output_schema=output_schema,
    )

    (udf_results, templates_mapping,) = convert_udfgen2nodeudf_results_and_mapping(
        udf_execution_queries, context_id, command_id
    )

    # Adding the udf_name and node_identifier to the mapping
    templates_mapping.update(
        {
            "udf_name": udf_name,
            "node_id": node_config.identifier,
            "min_row_count": node_config.privacy.minimum_row_count,
            "request_id": request_id,
        }
    )
    udf_statements = _create_udf_statements(udf_execution_queries, templates_mapping)

    return udf_statements, udf_results
