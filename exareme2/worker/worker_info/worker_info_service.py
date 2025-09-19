from exareme2.worker import config as worker_config
from exareme2.worker.utils.logger import initialise_logger
from exareme2.worker.worker_info import worker_info_db
from exareme2.worker.worker_info.worker_info_db import check_database_connection
from exareme2.worker.worker_info.worker_info_db import get_data_models
from exareme2.worker.worker_info.worker_info_db import get_dataset_infos
from exareme2.worker_communication import CommonDataElements
from exareme2.worker_communication import DataModelAttributes
from exareme2.worker_communication import DatasetsInfoPerDataModel
from exareme2.worker_communication import MonetDBConfig
from exareme2.worker_communication import WorkerInfo


@initialise_logger
def get_worker_info(request_id: str) -> WorkerInfo:
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging
    """

    monetdb_configs = (
        MonetDBConfig(port=worker_config.monetdb.port, ip=worker_config.monetdb.ip)
        if worker_config.monetdb.enabled
        else None
    )
    return WorkerInfo(
        id=worker_config.identifier,
        role=worker_config.role,
        ip=worker_config.rabbitmq.ip,
        port=worker_config.rabbitmq.port,
        monetdb_configs=monetdb_configs,
    )


@initialise_logger
def get_worker_datasets_per_data_model(request_id: str) -> DatasetsInfoPerDataModel:
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging
    Returns
    ------
    DatasetsInfoPerDataModel
        A dictionary with key data model and value  a dictionary with keys dataset and value each corresponding Info (label)
    """
    return DatasetsInfoPerDataModel(
        datasets_info_per_data_model={
            data_model: get_dataset_infos(data_model)
            for data_model in get_data_models()
        }
    )


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
    return worker_info_db.get_data_model_attributes(data_model)


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
    return worker_info_db.get_data_model_cdes(data_model)


@initialise_logger
def healthcheck(request_id: str, check_db):
    """
    If the check_db flag is false then the only purpose of the healthcheck method is to ensure that the WORKER service
    properly receives a task and responds.

    Parameters
    ----------
    request_id : str
        The identifier for the logging
    check_db : str
        Should also check the database health?
    """
    if check_db:
        check_database_connection()
