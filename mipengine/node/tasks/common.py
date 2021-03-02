from celery import shared_task

from mipengine.node.monetdb_interface import common_action
from mipengine.common.node_tasks_DTOs import TableData


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
    schema = common_action.get_table_schema(table_name)
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
    schema = common_action.get_table_schema(table_name)
    data = common_action.get_table_data(table_name)
    return TableData(schema, data).to_json()


@shared_task
def clean_up(context_id: str):
    common_action.clean_up(context_id)
