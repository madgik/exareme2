from typing import List

from celery import shared_task

from mipengine.node import DATA_TABLE_PRIMARY_KEY
from mipengine.node import config as node_config
from mipengine.node.monetdb_interface import views
from mipengine.node.monetdb_interface.common_actions import create_table_name
from mipengine.node.monetdb_interface.common_actions import get_data_model_datasets
from mipengine.node.monetdb_interface.common_actions import get_data_models
from mipengine.node.node_logger import initialise_logger
from mipengine.node_exceptions import DataModelUnavailable
from mipengine.node_exceptions import DatasetUnavailable
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
def create_data_model_view(
    request_id: str,
    context_id: str,
    command_id: str,
    data_model: str,
    datasets: List[str],
    columns: List[str],
    filters: dict = None,
) -> str:
    """
    Creates a MIP specific view of a data_model with specific columns, filters and datasets to the DB.

    Parameters
    ----------
    request_id : str
        The identifier for the logging
    context_id : str
        The id of the experiment
    command_id : str
        The id of the command that the view
    data_model : str
        The data_model data table on which the view will be created
    datasets : List[str]
        The datasets that will be used in the view.
    columns : List[str]
        A list of column names
    filters : dict
        A Jquery filters object

    Returns
    ------
    str
        The name of the created view
    """
    validate_data_model_and_datasets_exist(data_model, datasets)

    view_name = create_table_name(
        TableType.VIEW,
        node_config.identifier,
        context_id,
        command_id,
    )
    columns.insert(0, DATA_TABLE_PRIMARY_KEY)

    views.create_view(
        view_name=view_name,
        table_name=f'"{data_model}"."primary_data"',
        columns=columns,
        filters=filters,
        enable_min_rows_threshold=True,
    )
    return view_name


def validate_data_model_and_datasets_exist(data_model: str, datasets: List[str]):
    if data_model not in get_data_models():
        raise DataModelUnavailable(node_config.identifier, data_model)

    available_datasets = get_data_model_datasets(data_model)
    for dataset in datasets:
        if dataset not in available_datasets:
            raise DatasetUnavailable(node_config.identifier, dataset)


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
