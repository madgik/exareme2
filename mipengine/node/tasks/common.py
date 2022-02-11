from celery import shared_task

from mipengine.node import config as node_config
from mipengine.node.monetdb_interface import common_actions
from mipengine.node.monetdb_interface.common_actions import get_initial_data_models
from mipengine.node.monetdb_interface.common_actions import get_data_model_datasets
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
    datasets_per_data_model_code = {}
    for data_model_code in get_initial_data_models():
        datasets_per_data_model_code[data_model_code] = get_data_model_datasets(
            data_model_code
        )

    node_info = NodeInfo(
        id=node_config.identifier,
        role=node_config.role,
        ip=node_config.rabbitmq.ip,
        port=node_config.rabbitmq.port,
        db_ip=node_config.monetdb.ip,
        db_port=node_config.monetdb.port,
        datasets_per_data_model_code=datasets_per_data_model_code,
    )

    return node_info.json()


@shared_task
@initialise_logger
def get_table_schema(request_id: str, table_name: str) -> str:
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging
    table_name : str
        The name of the table

    Returns
    ------
    str(TableSchema)
        A TableSchema object in a jsonified format
    """
    schema = common_actions.get_table_schema(table_name)
    return schema.json()


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
def clean_up(request_id: str, context_id: str):
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging
    context_id : str
        The id of the experiment
    """
    common_actions.drop_db_artifacts_by_context_id(context_id)
