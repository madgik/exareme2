from typing import List

from exareme2 import DATA_TABLE_PRIMARY_KEY
from exareme2.worker import config as worker_config
from exareme2.worker.exareme2.tables.tables_db import create_table_name
from exareme2.worker.exareme2.views import views_db
from exareme2.worker.utils.logger import initialise_logger
from exareme2.worker.worker_info.worker_info_db import get_data_models
from exareme2.worker.worker_info.worker_info_db import get_dataset_infos
from exareme2.worker_communication import DataModelUnavailable
from exareme2.worker_communication import DatasetUnavailable
from exareme2.worker_communication import TableInfo
from exareme2.worker_communication import TableType

MINIMUM_ROW_COUNT = worker_config.privacy.minimum_row_count


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
    return views_db.get_view_names(context_id)


@initialise_logger
def create_data_model_views(
    request_id: str,
    context_id: str,
    command_id: str,
    data_model: str,
    datasets: List[str],
    columns_per_view: List[List[str]],
    filters: dict = None,
    dropna: bool = True,
    check_min_rows: bool = True,
) -> List[TableInfo]:
    """
    Create a view on a provided data model with specific columns, filters and datasets to the DB.

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
    columns_per_view : List[List[str]]
        One view will be created for each list of columns.
    filters : dict
        A Jquery filters object
    dropna : bool
        A flag that determines if the not null constraints about the columns should be included in the filters
    check_min_rows : bool
        A flag that determines if the min_rows_threshold should be checked.
    """
    _validate_data_model_and_datasets_exist(data_model, datasets)
    if datasets:
        filters = _get_filters_with_datasets_constraints(
            filters=filters, datasets=datasets
        )

    # In each view, it's not null constraints should include the columns of ALL views
    if dropna:
        all_columns = [column for columns in columns_per_view for column in columns]
        filters = _get_filters_with_columns_not_null_constraints(
            filters=filters, columns=all_columns
        )

    return [
        create_data_model_view(
            context_id=context_id,
            command_id=command_id,
            result_id=str(count),
            data_model=data_model,
            columns=view_columns,
            filters=filters,
            check_min_rows=check_min_rows,
        )
        for count, view_columns in enumerate(columns_per_view)
    ]


def create_data_model_view(
    context_id: str,
    command_id: str,
    result_id: str,
    data_model: str,
    columns: List[str],
    filters: dict = None,
    check_min_rows: bool = True,
) -> TableInfo:
    view_name = create_table_name(
        table_type=TableType.VIEW,
        worker_id=worker_config.identifier,
        context_id=context_id,
        command_id=command_id,
        result_id=result_id,
    )
    columns.insert(0, DATA_TABLE_PRIMARY_KEY)

    return views_db.create_view(
        view_name=view_name,
        table_name=f'"{data_model}"."primary_data"',
        columns=columns,
        filters=filters,
        minimum_row_count=MINIMUM_ROW_COUNT,
        check_min_rows=check_min_rows,
    )


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


def _get_filters_with_columns_not_null_constraints(filters, columns):
    """
    This function will return the given filters which will also include the columns' not null constraints.
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
        raise DataModelUnavailable(worker_config.identifier, data_model)

    available_dataset_infos = get_dataset_infos(data_model)
    available_datasets = [dataset_info.code for dataset_info in available_dataset_infos]
    for dataset in datasets:
        if dataset not in available_datasets:
            raise DatasetUnavailable(worker_config.identifier, dataset)


@initialise_logger
def create_view(
    request_id: str,
    context_id: str,
    command_id: str,
    table_name: str,
    columns: List[str],
    filters: dict,
) -> TableInfo:
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
    """
    view_name = create_table_name(
        TableType.VIEW,
        worker_config.identifier,
        context_id,
        command_id,
    )
    return views_db.create_view(
        view_name=view_name,
        table_name=table_name,
        columns=columns,
        filters=filters,
        minimum_row_count=MINIMUM_ROW_COUNT,
    )
