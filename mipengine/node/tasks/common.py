from celery import shared_task

from mipengine.node import config as node_config
from mipengine.node.monetdb_interface import common_actions
from mipengine.node.monetdb_interface.common_actions import get_initial_data_schemas
from mipengine.node.monetdb_interface.common_actions import get_schema_datasets
from mipengine.node_info_DTOs import NodeInfo
from mipengine.node_tasks_DTOs import TableData
from mipengine.node import logging_module as logging
from mipengine.node.logging_module import log_function_call

@shared_task
@log_function_call
def get_node_info():
    """
    Returns
    ------
    str(NodeInfo)
        A NodeInfo object in a jsonified format
    """
    datasets_per_schema = {}
    for schema in get_initial_data_schemas():
        datasets_per_schema[schema] = get_schema_datasets(schema)

    node_info = NodeInfo(
        id=node_config.identifier,
        role=node_config.role,
        ip=node_config.rabbitmq.ip,
        port=node_config.rabbitmq.port,
        db_ip=node_config.monetdb.ip,
        db_port=node_config.monetdb.port,
        datasets_per_schema=datasets_per_schema,
    )

    return node_info.json()


@shared_task
@log_function_call
def get_table_schema(table_name: str) -> str:
    """
    Parameters
    ----------
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
@log_function_call
def get_table_data(table_name: str) -> str:
    """
    Parameters
    ----------
    table_name : str
        The name of the table

    Returns
    ------
    str(TableData)
        An object of TableData in a jsonified format
    """
    schema = common_actions.get_table_schema(table_name)
    data = common_actions.get_table_data(table_name)
    return TableData(schema_=schema, data_=data).json()


@shared_task
@log_function_call
def clean_up(context_id: str):
    """
    Parameters
    ----------
    context_id : str
        The id of the experiment
    """
    common_actions.drop_db_artifacts_by_context_id(context_id)
