from typing import List

from celery import shared_task

from mipengine.node.monetdb_interface import common
from mipengine.node.monetdb_interface import merge_tables
from mipengine.node.monetdb_interface.common import config
from mipengine.node.monetdb_interface.common import create_table_name
from mipengine.node.monetdb_interface.merge_tables import get_type_of_tables
from mipengine.node.tasks.data_classes import TableInfo


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
            A list of merged table names
    """
    return merge_tables.get_merge_tables_names(context_id)


@shared_task
def create_merge_table(context_id: str, command_id: str, partition_table_names: List[str]) -> str:
    """
        Parameters
        ----------
        context_id : str
            The id of the experiment
        command_id : str
            The id of the command that the merge table
        partition_table_names: List[str]
            Its a list of names of the tables to be merged

        Returns
        ------
        str
            The name(string) of the created merge table in lower case.
    """
    get_type_of_tables(partition_table_names)
    schema = common.get_table_schema(partition_table_names[0])
    merge_table_name = create_table_name("merge", command_id, context_id, config["node"]["identifier"])
    table_info = TableInfo(merge_table_name.lower(), schema)
    merge_tables.create_merge_table(table_info)
    merge_tables.add_to_merge_table(merge_table_name, partition_table_names)
    return merge_table_name.lower()
