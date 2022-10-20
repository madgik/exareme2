from typing import Dict

from celery import shared_task

from mipengine.node import config as node_config
from mipengine.node.monetdb_interface import common_actions
from mipengine.node.monetdb_interface.common_actions import get_data_models
from mipengine.node.monetdb_interface.common_actions import (
    get_dataset_code_per_dataset_label,
)
from mipengine.node.node_logger import initialise_logger
from mipengine.node_info_DTOs import NodeInfo
from mipengine.node_tasks_DTOs import TableData


@shared_task
@initialise_logger
def get_node_info(request_id: str):
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging
    Returns
    ------
    str(NodeInfo)
        A NodeInfo object in a jsonified format
    """

    node_info = NodeInfo(
        id=node_config.identifier,
        role=node_config.role,
        ip=node_config.rabbitmq.ip,
        port=node_config.rabbitmq.port,
        db_ip=node_config.monetdb.ip,
        db_port=node_config.monetdb.port,
    )

    return node_info.json()


@shared_task
@initialise_logger
def get_node_datasets_per_data_model(request_id: str) -> Dict[str, Dict[str, str]]:
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging
    Returns
    ------
    Dict[str, Dict[str, str]]
        A dictionary with key data model and value a list of pairs (dataset code and dataset label)
    """
    return {
        data_model: get_dataset_code_per_dataset_label(data_model)
        for data_model in get_data_models()
    }


@shared_task
@initialise_logger
def get_data_model_attributes(request_id: str, data_model: str) -> str:
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging
    data_model: str
        The data model to retrieve its attributes.

    Returns
    ------
    Properties
        A DataModelAttributes object in a jsonified format
    """
    return common_actions.get_data_model_attributes(data_model).json()


@shared_task
@initialise_logger
def get_data_model_cdes(request_id: str, data_model: str) -> str:
    """
    Parameters
    ----------
    request_id: str
        The identifier for the logging
    data_model: str
        The data model to retrieve it's cdes.

    Returns
    ------
    str
        A CommonDataElements object in a jsonified format
    """
    return common_actions.get_data_model_cdes(data_model).json()


@shared_task
@initialise_logger
def get_table_data(request_id: str, table_name: str) -> str:
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging
    table_name : str
        The name of the table

    Returns
    ------
    str(TableData)
        An object of TableData in a jsonified format
    """
    columns = common_actions.get_table_data(table_name)

    return TableData(name=table_name, columns=columns).json()


@shared_task
@initialise_logger
def cleanup(request_id: str, context_id: str):
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging
    context_id : str
        The id of the experiment
    """
    common_actions.drop_db_artifacts_by_context_id(context_id)
