from typing import List

from celery import shared_task

from mipengine.node.monetdb_interface import views
from mipengine.node import config as node_config
from mipengine.node import DATA_TABLE_PRIMARY_KEY

from mipengine.node.monetdb_interface.common_actions import create_table_name


@shared_task
def get_views(context_id: str) -> List[str]:
    """
    Parameters
    ----------
    context_id : str
        The id of the experiment

    Returns
    ------
    List[str]
        A list of view names
    """
    return views.get_view_names(context_id)


@shared_task
def create_pathology_view(
        context_id: str,
        command_id: str,
        pathology: str,
        datasets: List[str],
        columns: List[str],
        filters: str = None,
) -> str:
    """
    Creates a MIP specific view of a pathology with specific columns, filters and datasets to the DB.

    Parameters
    ----------
    context_id : str
        The id of the experiment
    command_id : str
        The id of the command that the view
    pathology : str
        The pathology data table on which the view will be created
    datasets : List[str]
        A list of dataset names
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
        "view", command_id, context_id, node_config.identifier
    )
    columns.insert(0, DATA_TABLE_PRIMARY_KEY)

    filter_with_datasets = update_filters_with_datasets(filters, datasets)

    views.create_view(
        view_name=view_name,
        table_name=f"{pathology}_data",
        columns=columns,
        filters=filter_with_datasets,
    )
    return view_name


@shared_task
def create_view(
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
        "view", command_id, context_id, node_config.identifier
    )
    views.create_view(
        view_name=view_name,
        table_name=table_name,
        columns=columns,
        filters=filters,
    )
    return view_name


def update_filters_with_datasets(filters, datasets):
    rules = [
        {
            "id": "dataset",
            "field": "dataset",
            "type": "string",
            "input": "text",
            "operator": "in",
            "value": datasets,
        }]

    if filters is not None:
        rules.append(filters)
    return {
        "condition": "AND",
        "rules": rules,
        "valid": True,
    }
