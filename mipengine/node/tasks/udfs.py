import inspect
from typing import Dict
from typing import List
from typing import Tuple

from celery import shared_task

from mipengine import algorithms  # DO NOT REMOVE, NEEDED FOR ALGORITHM IMPORT

from mipengine.node import config as node_config
from mipengine.node.monetdb_interface import udfs
from mipengine.node.monetdb_interface.common_actions import create_table_name
from mipengine.node.monetdb_interface.common_actions import get_table_schema
from mipengine.node.monetdb_interface.common_actions import get_table_type
from mipengine.node_tasks_DTOs import TableInfo
from mipengine.node_tasks_DTOs import TableType
from mipengine.node_tasks_DTOs import UDFArgument
from mipengine.node_tasks_DTOs import UDFArgumentKind
from mipengine.udfgen import generate_udf_queries


@shared_task
def get_udfs(algorithm_name: str) -> List[str]:
    return [
        inspect.getsource(udf)
        for udf_name, udf in UDF_REGISTRY.items()
        if udf_name.startswith(algorithm_name)
    ]


# TODO Verify time limit when udf tests are fixed
@shared_task(
    soft_time_limit=node_config.celery.run_udf_soft_time_limit,
    time_limit=node_config.celery.run_udf_time_limit,
)
def run_udf(
    command_id: str,
    context_id: str,
    func_name: str,
    positional_args_json: List[str],
    keyword_args_json: Dict[str, str],
) -> str:
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

    Returns
    -------
        str
            The name of the table where the udf execution results are in.
    """

    positional_args = [UDFArgument.parse_raw(arg) for arg in positional_args_json]

    keyword_args = {
        key: UDFArgument.parse_raw(arg) for key, arg in keyword_args_json.items()
    }

    udf_creation_stmt, udf_execution_stmt, result_table_name = _generate_udf_statements(
        command_id, context_id, func_name, positional_args, keyword_args
    )

    udfs.run_udf(udf_creation_stmt, udf_execution_stmt)

    return result_table_name


@shared_task
def get_run_udf_query(
    command_id: str,
    context_id: str,
    func_name: str,
    positional_args_json: List[str],
    keyword_args_json: Dict[str, str],
) -> Tuple[str, str, str]:
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

    Returns
    -------
        str
            The name of the result table,
            the statement that creates the udf and
            the statement that executes the udf.
    """

    positional_args = [UDFArgument.parse_raw(arg) for arg in positional_args_json]

    keyword_args = {
        key: UDFArgument.parse_raw(arg) for key, arg in keyword_args_json.items()
    }

    return _generate_udf_statements(
        command_id, context_id, func_name, positional_args, keyword_args
    )


def _create_udf_name(func_name: str, command_id: str, context_id: str) -> str:
    """
    Creates a udf name with the format <func_name>_<commandId>_<contextId>
    """
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
):
    allowed_func_name = func_name.replace(".", "_")  # A dot is not an allowed character
    udf_name = _create_udf_name(allowed_func_name, command_id, context_id)
    result_table_name = create_table_name(
        TableType.NORMAL, command_id, context_id, node_config.identifier
    )

    gen_pos_args, gen_kw_args = _convert_udf2udfgen_args(positional_args, keyword_args)
    udf_creation_stmt, udf_execution_stmt = generate_udf_queries(
        func_name, gen_pos_args, gen_kw_args
    )
    udf_creation_stmt = udf_creation_stmt.substitute(udf_name=udf_name)
    udf_execution_stmt = udf_execution_stmt.substitute(
        table_name=result_table_name,
        udf_name=udf_name,
        node_id=node_config.identifier,
    )

    return udf_creation_stmt, udf_execution_stmt, result_table_name
