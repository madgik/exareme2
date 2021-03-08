from typing import List

from celery import shared_task

from mipengine.common.node_tasks_DTOs import TableInfo
from mipengine.node.monetdb_interface import remote_tables
from mipengine.node.monetdb_interface.connection_pool import get_connection, release_connection


@shared_task
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
    connection = get_connection()
    remote_table_names = remote_tables.get_remote_tables_names(connection.cursor(), context_id)
    release_connection(connection)
    return remote_table_names


@shared_task
def create_remote_table(table_info_json: str, url: str):
    """
        Parameters
        ----------
        table_info_json : str(TableInfo)
            A TableInfo object in a jsonified format
        url : str
            The url of the monetdb that we want to create the remote table from.
    """
    connection = get_connection()
    table_info = TableInfo.from_json(table_info_json)
    remote_tables.create_remote_table(connection.cursor(), table_info, url)
    connection.commit()
    release_connection(connection)
