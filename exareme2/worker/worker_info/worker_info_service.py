from typing import Dict
from typing import List
from typing import Tuple

from exareme2.worker import config as worker_config
from exareme2.worker.utils.logger import initialise_logger
from exareme2.worker.worker_info import worker_info_db
from exareme2.worker.worker_info.worker_info_db import check_database_connection
from exareme2.worker.worker_info.worker_info_db import get_data_models
from exareme2.worker.worker_info.worker_info_db import (
    get_dataset_code_per_dataset_label,
)
from exareme2.worker.worker_info.worker_info_db import get_datasets_per_data_model
from exareme2.worker_communication import DataModelMetadata
from exareme2.worker_communication import WorkerInfo


@initialise_logger
def get_worker_info(request_id: str) -> WorkerInfo:
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging
    """

    return WorkerInfo(
        id=worker_config.identifier,
        role=worker_config.role,
        ip=worker_config.rabbitmq.ip,
        port=worker_config.rabbitmq.port,
        db_ip=worker_config.monetdb.ip,
        db_port=worker_config.monetdb.port,
    )


@initialise_logger
def get_worker_data_model_metadata_and_datasets(
    request_id: str,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
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
    return worker_info_db.get_data_model_metadata(), get_datasets_per_data_model()


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
