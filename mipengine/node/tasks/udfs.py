from typing import Dict
from typing import List
from typing import Tuple

from celery import shared_task

from mipengine.node import config as node_config
from mipengine.node.monetdb_interface import udfs
from mipengine.node.monetdb_interface.common_actions import create_table_name
from mipengine.node.monetdb_interface.common_actions import get_table_schema
from mipengine.node.monetdb_interface.common_actions import get_table_type
from mipengine.node.node_logger import initialise_logger
from mipengine.node_exceptions import SMPCUsageError
from mipengine.node_tasks_DTOs import NodeLiteralDTO
from mipengine.node_tasks_DTOs import NodeSMPCDTO
from mipengine.node_tasks_DTOs import NodeSMPCValueDTO
from mipengine.node_tasks_DTOs import NodeTableDTO
from mipengine.node_tasks_DTOs import NodeUDFDTO
from mipengine.node_tasks_DTOs import TableInfo
from mipengine.node_tasks_DTOs import TableType
from mipengine.node_tasks_DTOs import UDFKeyArguments
from mipengine.node_tasks_DTOs import UDFPosArguments
from mipengine.node_tasks_DTOs import UDFResults
from mipengine.node_tasks_DTOs import _NodeUDFDTOType
from mipengine.udfgen import generate_udf_queries
from mipengine.udfgen.udfgen_DTOs import SMPCTablesInfo
from mipengine.udfgen.udfgen_DTOs import SMPCUDFGenResult
from mipengine.udfgen.udfgen_DTOs import TableUDFGenResult
from mipengine.udfgen.udfgen_DTOs import UDFGenExecutionQueries
from mipengine.udfgen.udfgen_DTOs import UDFGenResult
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
        request_id: str
            The experiment identifier, common among all experiment related actions.
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
        str(UDFResults)
            The results, with the tablenames, that the execution created.
    """
    _validate_smpc_usage(use_smpc)

    positional_args = UDFPosArguments.parse_raw(positional_args_json)
    keyword_args = UDFKeyArguments.parse_raw(keyword_args_json)

    udf_statements, udf_results = _generate_udf_statements(
        command_id=command_id,
        context_id=context_id,
        func_name=func_name,
        positional_args=positional_args,
        keyword_args=keyword_args,
        use_smpc=use_smpc,
    )

    udfs.run_udf(udf_statements)

    return udf_results.json()


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
    _validate_smpc_usage(use_smpc)

    positional_args = UDFPosArguments.parse_raw(positional_args_json)
    keyword_args = UDFKeyArguments.parse_raw(keyword_args_json)

    udf_statements, _ = _generate_udf_statements(
        command_id=command_id,
        context_id=context_id,
        func_name=func_name,
        positional_args=positional_args,
        keyword_args=keyword_args,
        use_smpc=use_smpc,
    )

    return udf_statements


def _validate_smpc_usage(use_smpc: bool):
    """
    Validates if smpc can be used or if it must be used based on the NODE configs.
    """
    if use_smpc and not node_config.smpc.enabled:
        raise SMPCUsageError("SMPC cannot be used, since it's not enabled on the node.")

    if not use_smpc and node_config.smpc.enabled and not node_config.smpc.optional:
        raise SMPCUsageError(
            "The computation cannot be made without SMPC. SMPC usage is not optional."
        )


def _create_udf_name(func_name: str, command_id: str, context_id: str) -> str:
    """
    Creates a udf name with the format <func_name>_<commandId>_<contextId>
    """
    # TODO Monetdb UDF name cannot be larger than 63 character
    return f"{func_name}_{command_id}_{context_id}"


def _create_table_info_from_tablename(tablename: str):
    return TableInfo(
        name=tablename,
        schema_=get_table_schema(tablename),
        type_=get_table_type(tablename),
    )


def _convert_smpc_udf2udfgen_arg(udf_argument: NodeSMPCDTO):
    template = _create_table_info_from_tablename(udf_argument.value.template.value)
    add_op = (
        _create_table_info_from_tablename(udf_argument.value.add_op_values.value)
        if udf_argument.value.add_op_values
        else None
    )
    min_op = (
        _create_table_info_from_tablename(udf_argument.value.min_op_values.value)
        if udf_argument.value.min_op_values
        else None
    )
    max_op = (
        _create_table_info_from_tablename(udf_argument.value.max_op_values.value)
        if udf_argument.value.max_op_values
        else None
    )
    union_op = (
        _create_table_info_from_tablename(udf_argument.value.union_op_values.value)
        if udf_argument.value.union_op_values
        else None
    )
    return SMPCTablesInfo(
        template=template,
        add_op_values=add_op,
        min_op_values=min_op,
        max_op_values=max_op,
        union_op_values=union_op,
    )


def _convert_udf2udfgen_arg(udf_argument: NodeUDFDTO):
    if isinstance(udf_argument, NodeLiteralDTO):
        return udf_argument.value
    elif isinstance(udf_argument, NodeTableDTO):
        return _create_table_info_from_tablename(udf_argument.value)
    elif isinstance(udf_argument, NodeSMPCDTO):
        return _convert_smpc_udf2udfgen_arg(udf_argument)
    else:
        argument_kinds = ",".join([str(k) for k in _NodeUDFDTOType])
        raise ValueError(
            f"A udf argument can have one of the following types {argument_kinds}'."
        )


def _convert_udf2udfgen_args(
    positional_args: UDFPosArguments,
    keyword_args: UDFKeyArguments,
):
    """
    The input arguments are received from the controller and are
    containing the value of the argument (literal) or information
    about the location of the input (tablename).

    This method adds information on these arguments, that will be
    sent to the udfgenerator.

    The information, such as TableType, TableSchema, can only be
    added from the NODE, not the controller, for security reasons.

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
        _convert_udf2udfgen_arg(pos_arg) for pos_arg in positional_args.args
    ]

    generator_kw_args = {
        key: _convert_udf2udfgen_arg(argument)
        for key, argument in keyword_args.args.items()
    }

    return generator_pos_args, generator_kw_args


def _convert_table_result_to_udf_statements(
    result: TableUDFGenResult, templates_mapping: dict
) -> List[str]:
    return [
        result.drop_query.substitute(**templates_mapping),
        result.create_query.substitute(**templates_mapping),
    ]


def _get_all_table_results_from_smpc_result(
    smpc_result: SMPCUDFGenResult,
) -> List[TableUDFGenResult]:
    table_results = [smpc_result.template]
    table_results.append(
        smpc_result.add_op_values
    ) if smpc_result.add_op_values else None
    table_results.append(
        smpc_result.min_op_values
    ) if smpc_result.min_op_values else None
    table_results.append(
        smpc_result.max_op_values
    ) if smpc_result.max_op_values else None
    table_results.append(
        smpc_result.union_op_values
    ) if smpc_result.union_op_values else None

    return table_results


def _convert_smpc_result_to_udf_statements(
    result: SMPCUDFGenResult, templates_mapping: dict
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
        if isinstance(result, TableUDFGenResult):
            udf_statements.extend(
                _convert_table_result_to_udf_statements(result, templates_mapping)
            )
        elif isinstance(result, SMPCUDFGenResult):
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


def _convert_udfgen2udf_table_result_and_mapping(
    udfgen_result: TableUDFGenResult,
    context_id: str,
    command_id: str,
    command_subid: int,
) -> Tuple[NodeTableDTO, Dict[str, str]]:
    table_name_ = create_table_name(
        table_type=TableType.NORMAL,
        node_id=node_config.identifier,
        context_id=context_id,
        command_id=command_id,
        command_subid=str(command_subid),
    )
    table_name_tmpl_mapping = {udfgen_result.tablename_placeholder: table_name_}
    return NodeTableDTO(value=table_name_), table_name_tmpl_mapping


def _convert_udfgen2udf_smpc_result_and_mapping(
    udfgen_result: SMPCUDFGenResult,
    context_id: str,
    command_id: str,
    command_subid: int,
) -> Tuple[NodeSMPCDTO, Dict[str, str]]:
    (
        template_udf_result,
        table_names_tmpl_mapping,
    ) = _convert_udfgen2udf_table_result_and_mapping(
        udfgen_result.template, context_id, command_id, command_subid
    )

    if udfgen_result.add_op_values:
        (add_op_udf_result, mapping,) = _convert_udfgen2udf_table_result_and_mapping(
            udfgen_result.add_op_values, context_id, command_id, command_subid + 1
        )
        table_names_tmpl_mapping.update(mapping)
    else:
        add_op_udf_result = None

    if udfgen_result.min_op_values:
        (min_op_udf_result, mapping,) = _convert_udfgen2udf_table_result_and_mapping(
            udfgen_result.min_op_values, context_id, command_id, command_subid + 2
        )
        table_names_tmpl_mapping.update(mapping)
    else:
        min_op_udf_result = None

    if udfgen_result.max_op_values:
        (max_op_udf_result, mapping,) = _convert_udfgen2udf_table_result_and_mapping(
            udfgen_result.max_op_values, context_id, command_id, command_subid + 3
        )
        table_names_tmpl_mapping.update(mapping)
    else:
        max_op_udf_result = None

    if udfgen_result.union_op_values:
        (union_op_udf_result, mapping,) = _convert_udfgen2udf_table_result_and_mapping(
            udfgen_result.union_op_values, context_id, command_id, command_subid + 4
        )
        table_names_tmpl_mapping.update(mapping)
    else:
        union_op_udf_result = None

    result = NodeSMPCDTO(
        value=NodeSMPCValueDTO(
            template=template_udf_result,
            add_op_values=add_op_udf_result,
            min_op_values=min_op_udf_result,
            max_op_values=max_op_udf_result,
            union_op_values=union_op_udf_result,
        )
    )
    return result, table_names_tmpl_mapping


def _convert_udfgen2udf_result_and_mapping(
    udfgen_result: UDFGenResult,
    context_id: str,
    command_id: str,
    command_subid: int,
) -> Tuple[NodeUDFDTO, Dict[str, str]]:
    if isinstance(udfgen_result, TableUDFGenResult):
        return _convert_udfgen2udf_table_result_and_mapping(
            udfgen_result, context_id, command_id, command_subid
        )
    elif isinstance(udfgen_result, SMPCUDFGenResult):
        return _convert_udfgen2udf_smpc_result_and_mapping(
            udfgen_result, context_id, command_id, command_subid
        )
    else:
        raise NotImplementedError


def convert_udfgen2udf_results_and_mapping(
    udf_queries: UDFGenExecutionQueries,
    context_id: str,
    command_id: str,
) -> Tuple[UDFResults, Dict[str, str]]:
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
        table_name, mapping = _convert_udfgen2udf_result_and_mapping(
            udf_result,
            context_id,
            command_id,
            command_subid,
        )
        table_names_tmpl_mapping.update(mapping)
        results.append(table_name)

        # Needs to be incremented by 10 because a udf_result could
        # contain more than one tables. (SMPC for example)
        command_subid += 10

    udf_results = UDFResults(results=results)
    return udf_results, table_names_tmpl_mapping


def _generate_udf_statements(
    command_id: str,
    context_id: str,
    func_name: str,
    positional_args: UDFPosArguments,
    keyword_args: UDFKeyArguments,
    use_smpc: bool,
) -> Tuple[List[str], UDFResults]:
    allowed_func_name = func_name.replace(".", "_")  # A dot is not an allowed character
    udf_name = _create_udf_name(allowed_func_name, command_id, context_id)

    gen_pos_args, gen_kw_args = _convert_udf2udfgen_args(positional_args, keyword_args)

    udf_execution_queries = generate_udf_queries(
        func_name, gen_pos_args, gen_kw_args, use_smpc
    )

    (udf_results, templates_mapping,) = convert_udfgen2udf_results_and_mapping(
        udf_execution_queries, context_id, command_id
    )

    # Adding the udf_name and node_identifier to the mapping
    templates_mapping.update(
        {
            "udf_name": udf_name,
            "node_id": node_config.identifier,
        }
    )
    udf_statements = _create_udf_statements(udf_execution_queries, templates_mapping)

    return udf_statements, udf_results
