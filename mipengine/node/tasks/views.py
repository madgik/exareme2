from typing import List

from celery import shared_task

from mipengine.node import DATA_TABLE_PRIMARY_KEY
from mipengine.node import config as node_config
from mipengine.node.monetdb_interface import views
from mipengine.node.monetdb_interface.common_actions import create_table_name
from mipengine.node.monetdb_interface.common_actions import get_data_models
from mipengine.node.monetdb_interface.common_actions import (
    get_dataset_code_per_dataset_label,
)
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
    dropna: bool = True,
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
    dropna : bool
        A bool that determines if the not null constraints about the columns should be included in the filters

    Returns
    ------
    str
        The name of the created view
    """
    _validate_data_model_and_datasets_exist(data_model, datasets)
    if datasets:
        filters = _get_filters_with_datasets_constraints(
            filters=filters, datasets=datasets
        )
    if not dropna:
        filters = _get_filters_with_columns_constraints(
            filters=filters, columns=columns
        )

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


def _get_filters_with_datasets_constraints(filters, datasets):
    """
    This function will return the given filters which will also include the dataset's constraints.
    """
    rules = [
        {
            "id": "dataset",
            "field": "dataset",
            "type": "string",
            "input": "text",
            "operator": "in",
            "value": datasets,
        }
    ]

    if filters:
        rules.append(filters)

    return {
        "condition": "AND",
        "rules": rules,
        "valid": True,
    }


def _get_filters_with_columns_constraints(filters, columns):
    """
    This function will return the given filters which will also include the column's constraints.
    """
    rules = [
        {
            "condition": "AND",
            "rules": [
                {
                    "id": variable,
                    "type": "string",
                    "operator": "is_not_null",
                    "value": None,
                }
                for variable in columns
            ],
        }
    ]

    if filters:
        rules.append(filters)

    return {
        "condition": "AND",
        "rules": rules,
        "valid": True,
    }


def _validate_data_model_and_datasets_exist(data_model: str, datasets: List[str]):
    if data_model not in get_data_models():
        raise DataModelUnavailable(node_config.identifier, data_model)

    available_datasets = get_dataset_code_per_dataset_label(data_model)
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
