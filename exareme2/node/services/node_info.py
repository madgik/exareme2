from typing import Dict

from exareme2.node import config as node_config
from exareme2.node.logger import initialise_logger
from exareme2.node.monetdb import node_info
from exareme2.node.monetdb.node_info import get_data_models
from exareme2.node.monetdb.node_info import get_dataset_code_per_dataset_label
from exareme2.node_communication import CommonDataElements
from exareme2.node_communication import DataModelAttributes
from exareme2.node_communication import NodeInfo


@initialise_logger
def get_node_info(request_id: str) -> NodeInfo:
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging
    """

    return NodeInfo(
        id=node_config.identifier,
        role=node_config.role,
        ip=node_config.rabbitmq.ip,
        port=node_config.rabbitmq.port,
        db_ip=node_config.monetdb.ip,
        db_port=node_config.monetdb.port,
    )


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


@initialise_logger
def get_data_model_attributes(request_id: str, data_model: str) -> DataModelAttributes:
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging
    data_model: str
        The data model to retrieve its attributes.
    """
    return node_info.get_data_model_attributes(data_model)


@initialise_logger
def get_data_model_cdes(request_id: str, data_model: str) -> CommonDataElements:
    """
    Parameters
    ----------
    request_id: str
        The identifier for the logging
    data_model: str
        The data model to retrieve it's cdes.
    """
    return node_info.get_data_model_cdes(data_model)
