from typing import List

from celery import shared_task

from mipengine.common.node_tasks_DTOs import TableInfo
from mipengine.common.node_tasks_DTOs import TableSchema
from mipengine.node.monetdb_interface import tables
from mipengine.node.monetdb_interface.common_actions import config
from mipengine.node.monetdb_interface.common_actions import create_table_name
from mipengine.node.monetdb_interface.connection_pool import get_connection, release_connection


@shared_task
def get_tables(context_id: str) -> List[str]:
    """
        Parameters
        ----------
        context_id : str
        The id of the experiment

        Returns
        ------
        List[str]
            A list of table names
    """
    connection = get_connection()
    cursor = connection.cursor()
    try:
        table_names = tables.get_tables_names(connection.cursor(), context_id)
        connection.commit()
        release_connection(connection, cursor)
    except Exception as exc:
        connection.rollback()
        release_connection(connection, cursor)
        raise exc
    return table_names


@shared_task
def create_table(context_id: str, command_id: str, schema_json: str) -> str:
    """
        Parameters
        ----------
        context_id : str
            The id of the experiment
        command_id : str
            The id of the command that the table
        schema_json : str(TableSchema)
            A TableSchema object in a jsonified format

        Returns
        ------
        str
            The name of the created table in lower case
    """
    schema_object = TableSchema.from_json(schema_json)
    table_name = create_table_name("table", command_id, context_id, config["node"]["identifier"])
    table_info = TableInfo(table_name.lower(), schema_object)
    connection = get_connection()
    cursor = connection.cursor()
    try:
        tables.create_table(connection, cursor, table_info)
        release_connection(connection, cursor)
    except Exception as exc:
        release_connection(connection, cursor)
        raise exc
    return table_name.lower()
