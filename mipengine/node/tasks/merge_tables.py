import json
from celery import shared_task

from mipengine.node.monetdb_interface import merge_tables
from mipengine.node.monetdb_interface import tables
from mipengine.node.monetdb_interface.common import config
from mipengine.node.monetdb_interface.common import create_table_name
from mipengine.node.tasks.data_classes import TableInfo


@shared_task
def get_merge_tables(context_id: str) -> str:
    """
        Parameters
        ----------
        context_id : str
            The id of the experiment

        Returns
        ------
        str --> (jsonified List[str])
            A list of merged table names in a jsonified format
    """
    return json.dumps(merge_tables.get_merge_tables_names(context_id))


# TODO Add in method description the jsonified input types
@shared_task
def create_merge_table(context_id: str, partition_table_names_json: str) -> str:
    """
        Parameters
        ----------
        context_id : str
            The id of the experiment
        partition_table_names_json : str --> (jsonified List[str])
            Its a list of names of the tables to be merged in a jsonified format

        Returns
        ------
        str
            The name(string) of the created merge table in lower case.
    """
    partition_table_names = json.loads(partition_table_names_json)
    merge_table_name = create_table_name("merge", context_id, config["node"]["identifier"])
    schema = tables.get_table_schema(partition_table_names[0])
    table_info = TableInfo(merge_table_name.lower(), schema)
    merge_tables.create_merge_table(table_info)
    merge_tables.add_to_merge_table(merge_table_name, partition_table_names)
    return merge_table_name.lower()
