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
from mipengine.node_tasks_DTOs import TableInfo
from mipengine.node_tasks_DTOs import TableType
from mipengine.node_tasks_DTOs import UDFArgument
from mipengine.node_tasks_DTOs import UDFArgumentKind
from mipengine.udfgen import generate_udf_queries
from mipengine.udfgen.udfgenerator import udf as udf_registry


@shared_task
@initialise_logger
def get_udf(func_name: str) -> str:
    return str(udf_registry.registry[func_name])


# TODO https://team-1617704806227.atlassian.net/browse/MIP-473
# @shared_task(
#     soft_time_limit=node_config.celery.run_udf_soft_time_limit,
#     time_limit=node_config.celery.run_udf_time_limit,
# )
@shared_task
@initialise_logger
def run_udf(
    command_id: str,
    context_id: str,
    func_name: str,
    positional_args_json: List[str],
    keyword_args_json: Dict[str, str],
    use_smpc: bool = False,
) -> List[str]:
    """
    Creates the UDF, if provided, and adds it in the database.
    Then it runs the select statement with the input provided.

    Parameters
    ----------
        command_id: str
            The command identifier, common among all nodes for this action.
        context_id: str
            The experiment identifier, common among all experiment related actions.
        func_name: str
            Name of function from which to generate UDF.
        positional_args_json: list[str(UDFArgument)]
            Positional arguments of the udf call.
        keyword_args_json: dict[str, str(UDFArgument)]
            Keyword arguments of the udf call.
        use_smpc: bool
            Should SMPC be used?
    Returns
    -------
        List[str]
            The names of the tables where the udf execution results are in.
    """
    _validate_smpc_usage(use_smpc)

    positional_args = [UDFArgument.parse_raw(arg) for arg in positional_args_json]

    keyword_args = {
        key: UDFArgument.parse_raw(arg) for key, arg in keyword_args_json.items()
    }

    udf_statements, result_table_names = _generate_udf_statements(
        command_id=command_id,
        context_id=context_id,
        func_name=func_name,
        positional_args=positional_args,
        keyword_args=keyword_args,
        use_smpc=use_smpc,
    )

    udfs.run_udf(udf_statements)

    return result_table_names


@shared_task
@initialise_logger
def get_run_udf_query(
    command_id: str,
    context_id: str,
    func_name: str,
    positional_args_json: List[str],
    keyword_args_json: Dict[str, str],
    use_smpc: bool = False,
) -> List[str]:
    """
    Fetches the sql statements that represent the execution of the udf.

    Parameters
    ----------
        command_id: str
            The command identifier, common among all nodes for this action.
        context_id: str
            The experiment identifier, common among all experiment related actions.
        func_name: str
            Name of function from which to generate UDF.
        positional_args_json: list[str(UDFArgument)]
            Positional arguments of the udf call.
        keyword_args_json: dict[str, str(UDFArgument)]
            Keyword arguments of the udf call.
        use_smpc: bool
            Should SMPC be used?
    Returns
    -------
        List[str]
            A list of the statements that would be executed in the DB.

    """
    _validate_smpc_usage(use_smpc)

    positional_args = [UDFArgument.parse_raw(arg) for arg in positional_args_json]

    keyword_args = {
        key: UDFArgument.parse_raw(arg) for key, arg in keyword_args_json.items()
    }

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


def _convert_udf2udfgen_arg(udf_argument: UDFArgument):
    if udf_argument.kind == UDFArgumentKind.LITERAL:
        return udf_argument.value
    elif udf_argument.kind == UDFArgumentKind.TABLE:
        return TableInfo(
            name=udf_argument.value,
            schema_=get_table_schema(udf_argument.value),
            type_=get_table_type(udf_argument.value),
        )
    else:
        argument_kinds = ",".join([str(k) for k in UDFArgumentKind])
        raise ValueError(
            f"A udf argument can have one of the following types {argument_kinds}'."
        )


def _convert_udf2udfgen_args(
    positional_args: List[UDFArgument],
    keyword_args: Dict[str, UDFArgument],
):
    generator_pos_args = [
        _convert_udf2udfgen_arg(pos_arg) for pos_arg in positional_args
    ]

    generator_kw_args = {
        key: _convert_udf2udfgen_arg(argument) for key, argument in keyword_args.items()
    }

    return generator_pos_args, generator_kw_args


def _generate_udf_statements(
    command_id: str,
    context_id: str,
    func_name: str,
    positional_args: List[UDFArgument],
    keyword_args: Dict[str, UDFArgument],
    use_smpc: bool,
) -> Tuple[List[str], List[str]]:
    allowed_func_name = func_name.replace(".", "_")  # A dot is not an allowed character
    udf_name = _create_udf_name(allowed_func_name, command_id, context_id)

    gen_pos_args, gen_kw_args = _convert_udf2udfgen_args(positional_args, keyword_args)

    udf_execution_queries = generate_udf_queries(
        func_name, gen_pos_args, gen_kw_args, use_smpc
    )

    result_tables = []
    output_table_names = {}
    for sequence, output_table in enumerate(udf_execution_queries.output_tables):
        table_name = create_table_name(
            table_type=TableType.NORMAL,
            node_id=node_config.identifier,
            context_id=context_id,
            command_id=command_id,
            command_subid=str(sequence),
        )
        output_table_names[output_table.tablename_placeholder] = table_name
        result_tables.append(table_name)

    templates_mapping = {
        "udf_name": udf_name,
        "node_id": node_config.identifier,
    }
    templates_mapping.update(output_table_names)

    udf_statements = []
    for output_table in udf_execution_queries.output_tables:
        udf_statements.append(output_table.drop_query.substitute(**templates_mapping))
        udf_statements.append(output_table.create_query.substitute(**templates_mapping))
    if udf_execution_queries.udf_definition_query:
        udf_statements.append(
            udf_execution_queries.udf_definition_query.substitute(**templates_mapping)
        )
    udf_statements.append(
        udf_execution_queries.udf_select_query.substitute(**templates_mapping)
    )

    return udf_statements, result_tables
