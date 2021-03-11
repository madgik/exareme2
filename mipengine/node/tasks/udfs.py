from typing import List, Dict

from celery import shared_task

from mipengine.common.node_tasks_DTOs import TableInfo
from mipengine.common.node_tasks_DTOs import UDFInfo
from mipengine.node.monetdb_interface import udfs
from mipengine.node.udfgen import generate_udf


@shared_task
def get_udfs() -> List[UDFInfo]:
    pass


@shared_task
def run_udf(func_name: str,
            positional_args: List[Dict],
            keyword_args: Dict[str, Dict],
            ) -> TableInfo:
    """
    Creates the udf provided with the given arguments and then runs it

    Parameters
    ----------
        func_name: str
            Name of function from which to generate UDF.
        udf_name: str
            Name to use in UDF definition.
        positional_args: list[dict]
            Positional parameter info objects.
        keyword_args: dict[str, dict]
            Keyword parameter info objects.

    Returns
    -------
        str
            Multiline string with MonetDB Python UDF definition.
    """
    # Add rows of table in the arguments
    # TODO jason needs to provide a clear way of doing that

    # TODO Create a udf naming convention
    udf_name = "test"

    # Generate the udf
    udf = generate_udf(
        func_name=func_name,
        udf_name=udf_name,
        positional_args=positional_args,
        keyword_args=keyword_args,
    )

    # Execute the udf in monet
    # TODO What is the input table?
    # TODO What is the output table name?
    udfs.run_udf("input_table_name",
                 udf,
                 "output_table_name")


@shared_task
def get_udf(udf_name: str) -> UDFInfo:
    pass
