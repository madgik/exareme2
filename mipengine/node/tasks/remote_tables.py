from typing import List

from celery import shared_task

from mipengine.node.monetdb_interface import remote_tables
from mipengine.node.node_logger import initialise_logger
from mipengine.node_tasks_DTOs import TableSchema


@shared_task
@initialise_logger
def get_remote_tables(context_id: str) -> List[str]:
    """
    Parameters
    ----------
    context_id : str
    The id of the experiment

    Returns
    ------
    List[str]
        A list of remote table names
    """
    return remote_tables.get_remote_table_names(context_id)


@shared_task
@initialise_logger
def create_remote_table(
    table_name: str, table_schema_json: str, monetdb_socket_address: str
):
    """
    Parameters
    ----------
    table_name : str
        The name of the table.
    table_schema : str(TableSchema)
        A TableSchema object in a jsonified format
    monetdb_socket_address : str
        The monetdb_socket_address of the monetdb that we want to create the remote table from.
    """
    schema = TableSchema.parse_raw(table_schema_json)
    remote_tables.create_remote_table(
        name=table_name, schema=schema, monetdb_socket_address=monetdb_socket_address
    )
