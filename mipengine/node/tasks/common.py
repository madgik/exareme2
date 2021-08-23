from celery import shared_task

from mipengine.node import config as node_config
from mipengine.node.monetdb_interface import common_actions
from mipengine.node_info_DTOs import NodeInfo
from mipengine.node_info_DTOs import NodeRole
from mipengine.node_tasks_DTOs import TableData


@shared_task
def get_node_info():
    """
    Returns
    ------
    str(NodeInfo)
        A NodeInfo object in a jsonified format
    """
    if node_config.role == NodeRole.LOCALNODE:
        datasets_per_schema = {
            "dementia": [
                "edsd",
                "ppmi",
                "desd-synthdata",
                "fake_longitudinal",
                "demo_data",
            ],
            "mentalhealth": ["demo"],
            "tbi": ["tbi_demo2"],
        }
    else:
        datasets_per_schema = None

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
    return schema.to_json()


@shared_task
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
    return TableData(schema, data).to_json()


@shared_task
def clean_up(context_id: str):
    """
    Parameters
    ----------
    context_id : str
        The id of the experiment
    """
    common_actions.drop_db_artifacts_by_context_id(context_id)
