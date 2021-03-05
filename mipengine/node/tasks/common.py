from celery import shared_task

from mipengine.node.monetdb_interface import common_actions
from mipengine.common.node_tasks_DTOs import TableData
from mipengine.node.monetdb_interface.common_actions import get_connection


@shared_task
def get_table_schema(table_name: str) -> str:
    """
        Parameters
        ----------
        table_name : str
        The name of the table

        Returns
        ------
        str(TableSchema)
            A TableSchema object in a jsonified format
    """
    connection = get_connection()
    cursor = connection.cursor()
    schema = common_actions.get_table_schema(cursor, table_name)
    return schema.to_json()


@shared_task
def get_table_data(table_name: str) -> str:
    """
        Parameters
        ----------
        table_name : str
            The name of the table

        Returns
        ------
        str(TableData)
            An object of TableData in a jsonified format
    """
    connection = get_connection()
    cursor = connection.cursor()
    schema = common_actions.get_table_schema(cursor, table_name)
    data = common_actions.get_table_data(cursor, table_name)
    return TableData(schema, data).to_json()


@shared_task
def clean_up(context_id: str):
    connection = get_connection()
    cursor = connection.cursor()
    common_actions.clean_up(cursor, context_id)
    connection.commit()
