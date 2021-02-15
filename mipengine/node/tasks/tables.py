import json
from typing import List

from celery import shared_task

from mipengine.node.monetdb_interface import tables
from mipengine.node.monetdb_interface.common import config
from mipengine.node.monetdb_interface.common import create_table_name
from mipengine.node.tasks.data_classes import ColumnInfo
from mipengine.node.tasks.data_classes import TableData
from mipengine.node.tasks.data_classes import TableInfo


@shared_task
def get_tables(context_id: str) -> List[str]:
    """
        Parameters
        ----------
        context_id : str
        The id of the experiment

        Returns
        ------
        str
            A list of table names in a jsonified format
    """
    return json.dumps(tables.get_tables_names(context_id))


@shared_task
def get_table_schema(table_name: str) -> List[ColumnInfo]:
    """
        Parameters
        ----------
        table_name : str
        The name of the table

        Returns
        ------
        A schema(list of ColumnInfo's objects) in a jsonified format
    """
    schema = tables.get_table_schema(table_name)
    return ColumnInfo.schema().dumps(schema, many=True)


@shared_task
def get_table_data(table_name: str) -> TableData:
    """
        Parameters
        ----------
        table_name : str
            The name of the table

        Returns
        ------
        str
            An object of TableData in a jsonified format
    """
    schema = tables.get_table_schema(table_name)
    data = tables.get_table_data(table_name)
    return TableData(schema, data).to_json()


@shared_task
def create_table(context_id: str, schema: str) -> str:
    """
        Parameters
        ----------
        context_id : str
            The id of the experiment
        schema : str
            A schema(list of ColumnInfo's objects) in a jsonified format

        Returns
        ------
        str
            The name of the created table in lower case
    """
    schema_object = ColumnInfo.schema().loads(schema, many=True)
    table_name = create_table_name("table", context_id, config["node"]["identifier"])
    table_info = TableInfo(table_name.lower(), schema_object)
    tables.create_table(table_info)
    return table_name.lower()
