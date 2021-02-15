import json
from typing import List

from celery import shared_task

from mipengine.node.monetdb_interface import remote_tables
from mipengine.node.tasks.data_classes import TableInfo


@shared_task
def get_remote_tables(context_id: str) -> List[str]:
    """
        Parameters
        ----------
        context_id : str
        The id of the experiment

        Returns
        ------
        str
            A list of remote table names in a jsonified format
    """
    return json.dumps(remote_tables.get_remote_tables_names(context_id))


@shared_task
def create_remote_table(table_info_json: str, url: str):
    """
        Parameters
        ----------
        table_info_json : str
            A TableInfo object in a jsonified format
        url : str
            The url of the monetdb that we want to create the remote table from.
    """
    table_info = TableInfo.from_json(table_info_json)
    remote_tables.create_remote_table(table_info, url)
