from typing import List

from celery import shared_task

from mipengine.node import config as node_config
from mipengine.node.monetdb_interface import merge_tables
from mipengine.node.monetdb_interface.common_actions import create_table_name
from mipengine.node.node_logger import initialise_logger
from mipengine.node_tasks_DTOs import TableInfo
from mipengine.node_tasks_DTOs import TableType


@shared_task
@initialise_logger
def get_merge_tables(request_id: str, context_id: str) -> List[str]:
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging
    context_id : str
        The id of the experiment

    Returns
    ------
    List[str]
        A list of merge table names
    """
    return merge_tables.get_merge_tables_names(context_id)


@shared_task
@initialise_logger
def create_merge_table(
    request_id: str, context_id: str, command_id: str, table_infos_json: List[str]
) -> str:
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging
    context_id : str
        The id of the experiment
    command_id : str
        The id of the command that the merge table
    table_infos_json: List[str(TableInfo)]
        A list of TableInfo of the tables to be merged, in a jsonified format

    Returns
    ------
    str(TableInfo)
        A TableInfo object in a jsonified format
    """
    table_infos = [
        TableInfo.parse_raw(table_info_json) for table_info_json in table_infos_json
    ]
    merge_table_name = create_table_name(
        TableType.MERGE,
        node_config.identifier,
        context_id,
        command_id,
    )

    merge_tables.create_merge_table(
        table_name=merge_table_name,
        table_schema=table_infos[0].schema_,
        merge_table_names=[table_info.name for table_info in table_infos],
    )

    return TableInfo(
        name=merge_table_name,
        schema_=table_infos[0].schema_,
        type_=TableType.MERGE,
    ).json()
