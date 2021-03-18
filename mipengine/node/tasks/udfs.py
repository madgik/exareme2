import inspect
from typing import List, Dict, Tuple

from celery import shared_task

from mipengine.algorithms import UDF_REGISTRY
from mipengine.algorithms import logistic_regression
from mipengine.algorithms import demo  # TODO Split the actual and testing algorithms
from mipengine.common.node_tasks_DTOs import UDFArgument
from mipengine.common.validate_identifier_names import validate_identifier_names
from mipengine.node.config.config_parser import config
from mipengine.node.monetdb_interface import udfs
from mipengine.node.monetdb_interface.common_action import get_table_schema, create_table_name
from mipengine.node.udfgen import ColumnInfo, TableInfo, generate_udf_application_queries


@shared_task
def get_udfs(algorithm_name: str) -> List[str]:
    return [inspect.getsource(udf) for udf_name, udf in UDF_REGISTRY.items()
            if udf_name.startswith(algorithm_name)]


@shared_task
def run_udf(command_id: str,
            context_id: str,
            func_name: str,
            positional_args_json: List[str],
            keyword_args_json: Dict[str, str],
            ) -> str:
    """
    Creates the UDF and adds it in the database.
    Then it runs the UDF with the input provided.

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

    result_table_name = create_table_name("table", command_id, context_id, config["node"]["identifier"])

    positional_args = [UDFArgument.from_json(arg)
                       for arg in positional_args_json]

    keyword_args = {key: UDFArgument.from_json(arg)
                    for key, arg in keyword_args_json.items()}

    udf_creation_stmt, udf_execution_stmt = __generate_udf_statements(command_id,
                                                                      context_id,
                                                                      func_name,
                                                                      positional_args,
                                                                      keyword_args)

    udfs.run_udf(udf_creation_stmt, udf_execution_stmt)

    return result_table_name


@shared_task
def get_run_udf_query(command_id: str,
                      context_id: str,
                      func_name: str,
                      positional_args_json: List[str],
                      keyword_args_json: Dict[str, str],
                      ) -> Tuple[str, str]:
    """
    Creates the UDF and adds it in the database.
    Then it runs the UDF with the input provided.

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

    positional_args = [UDFArgument.from_json(arg)
                       for arg in positional_args_json]

    keyword_args = {key: UDFArgument.from_json(arg)
                    for key, arg in keyword_args_json.items()}

    return __generate_udf_statements(command_id,
                                     context_id,
                                     func_name,
                                     positional_args,
                                     keyword_args)


@validate_identifier_names
def __create_udf_name(func_name: str, command_id: str, context_id: str) -> str:
    """
    Creates a udf name with the format <func_name>_<commandId>_<contextId>
    """
    return f"{func_name}_{command_id}_{context_id}"


def __convert_udf_arg_to_udfgen_arg(udf_argument: UDFArgument):
    if udf_argument.type == "literal":
        return udf_argument.value
    elif udf_argument.type == "table":
        name = udf_argument.value
        schema = get_table_schema(udf_argument.value)
        udf_generator_schema = [ColumnInfo(column.name, column.data_type)
                                for column in schema.columns]
        return TableInfo(name, udf_generator_schema)
    else:
        raise ValueError("A udf argument can have one of the following types 'literal','table'.")


def __convert_udf_args_to_udfgen_args(positional_args: List[UDFArgument],
                                      keyword_args: Dict[str, UDFArgument],
                                      ):
    generator_pos_args = [__convert_udf_arg_to_udfgen_arg(pos_arg)
                          for pos_arg in positional_args]

    generator_kw_args = {key: __convert_udf_arg_to_udfgen_arg(argument)
                         for key, argument in keyword_args.items()}

    return generator_pos_args, generator_kw_args


def __generate_udf_statements(command_id: str,
                              context_id: str,
                              func_name: str,
                              positional_args: List[UDFArgument],
                              keyword_args: Dict[str, UDFArgument],
                              ):
    gen_pos_args, gen_kw_args = __convert_udf_args_to_udfgen_args(positional_args, keyword_args)

    udf_creation_stmt, udf_execution_stmt = generate_udf_application_queries(func_name, gen_pos_args, gen_kw_args)

    allowed_func_name = func_name.replace('.', '_')  # A dot is not an allowed character
    udf_name = __create_udf_name(allowed_func_name, command_id, context_id)
    result_table_name = create_table_name("table", command_id, context_id, config["node"]["identifier"])
    udf_creation_stmt = udf_creation_stmt.substitute(udf_name=udf_name)
    udf_execution_stmt = udf_execution_stmt.substitute(table_name=result_table_name,
                                                       udf_name=udf_name,
                                                       node_id=config["node"]["identifier"])

    return udf_creation_stmt, udf_execution_stmt
