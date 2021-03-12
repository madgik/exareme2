from typing import List

from celery import shared_task

from mipengine.common.node_tasks_DTOs import TableInfo
from mipengine.node.monetdb_interface import common_actions
from mipengine.node.monetdb_interface import merge_tables
from mipengine.node.monetdb_interface.common_actions import config
from mipengine.node.monetdb_interface.common_actions import create_table_name
from mipengine.node.monetdb_interface.connection_pool import get_connection, release_connection
from mipengine.node.monetdb_interface.merge_tables import validate_tables_can_be_merged


@shared_task
def get_merge_tables(context_id: str) -> List[str]:
    """
        Parameters
        ----------
        context_id : str
            The id of the experiment

        Returns
        ------
        List[str]
            A list of merge table names
    """
    connection = get_connection()
    cursor = connection.cursor()
    try:
        merge_table_names = merge_tables.get_merge_tables_names(connection.cursor(), context_id)
        release_connection(connection, cursor)
    except Exception as exc:
        release_connection(connection, cursor)
        raise exc

    return merge_table_names


@shared_task
def create_merge_table(context_id: str, command_id: str, table_names: List[str]) -> str:
    """
        Parameters
        ----------
        context_id : str
            The id of the experiment
        command_id : str
            The id of the command that the merge table
        table_names: List[str]
            Its a list of names of the tables to be merged

        Returns
        ------
        str
            The name(string) of the created merge table in lower case.
    """
    connection = get_connection()
    cursor = connection.cursor()
    try:
        validate_tables_can_be_merged(cursor, table_names)
        schema = common_actions.get_table_schema(cursor, table_names[0])
        print(schema)
        merge_table_name = create_table_name("merge", command_id, context_id, config["node"]["identifier"])
        table_info = TableInfo(merge_table_name.lower(), schema)
        merge_tables.create_merge_table(connection, cursor, table_info)
        merge_tables.add_to_merge_table(connection, cursor, merge_table_name, table_names)
        release_connection(connection, cursor)
    except Exception as exc:
        release_connection(connection, cursor)
        raise exc

    return merge_table_name.lower()
