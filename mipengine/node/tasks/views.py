from typing import List

from celery import shared_task

from mipengine.node import DATA_TABLE_PRIMARY_KEY
from mipengine.node import config as node_config
from mipengine.node.monetdb_interface import views
from mipengine.node.monetdb_interface.common_actions import create_table_name
from mipengine.node.node_logger import initialise_logger
from mipengine.node_tasks_DTOs import TableType


@shared_task
@initialise_logger
def get_views(request_id: str, context_id: str) -> List[str]:
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
        A list of view names
    """
    return views.get_view_names(context_id)


@shared_task
@initialise_logger
def create_pathology_view(
    request_id: str,
    context_id: str,
    command_id: str,
    pathology: str,
    columns: List[str],
    filters: dict = None,
) -> str:
    """
    Creates a MIP specific view of a pathology with specific columns, filters and datasets to the DB.

    Parameters
    ----------
    request_id : str
        The identifier for the logging
    context_id : str
        The id of the experiment
    command_id : str
        The id of the command that the view
    pathology : str
        The pathology data table on which the view will be created
    columns : List[str]
        A list of column names
    filters : dict
        A Jquery filters object

    Returns
    ------
    str
        The name of the created view
    """
    view_name = create_table_name(
        TableType.VIEW,
        node_config.identifier,
        context_id,
        command_id,
    )
    columns.insert(0, DATA_TABLE_PRIMARY_KEY)

    # TODO Now the data_models require a version to access the proper table with data.
    views.create_view(
        view_name=view_name,
        table_name=f'"{pathology}:0.1"."primary_data"',
        columns=columns,
        filters=filters,
        enable_min_rows_threshold=True,
    )
    return view_name


@shared_task
@initialise_logger
def create_view(
    request_id: str,
    context_id: str,
    command_id: str,
    table_name: str,
    columns: List[str],
    filters: dict,
) -> str:
    """
    Creates a view of a table with specific columns and filters to the DB.

    Parameters
    ----------
    request_id : str
        The identifier for the logging
    context_id : str
        The id of the experiment
    command_id : str
        The id of the command that the view
    table_name : str
        The name of the table
    columns : List[str]
        A list of column names
    filters : dict
        A Jquery filters in a dict

    Returns
    ------
    str
        The name of the created view
    """
    view_name = create_table_name(
        TableType.VIEW,
        node_config.identifier,
        context_id,
        command_id,
    )
    views.create_view(
        view_name=view_name,
        table_name=table_name,
        columns=columns,
        filters=filters,
    )
    return view_name
