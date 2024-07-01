import json
from typing import Dict
from typing import List

from exareme2.worker.exareme2.monetdb.guard import is_datamodel
from exareme2.worker.exareme2.monetdb.guard import sql_injection_guard
from exareme2.worker.worker_info import sqlite
from exareme2.worker_communication import CommonDataElement
from exareme2.worker_communication import CommonDataElements
from exareme2.worker_communication import DataModelAttributes

HEALTHCHECK_VALIDATION_STRING = "HEALTHCHECK"


def get_data_models() -> List[str]:
    """
    Retrieves the enabled data_models from the database.

    Returns
    ------
    List[str]
        The data_models.
    """

    data_models_code_and_version = sqlite.execute_and_fetchall(
        f"""SELECT code, version
            FROM data_models
            WHERE status = 'ENABLED'
        """
    )
    data_models = [
        code + ":" + version for code, version in data_models_code_and_version
    ]
    return data_models


@sql_injection_guard(data_model=is_datamodel)
def get_dataset_code_per_dataset_label(data_model: str) -> Dict[str, str]:
    """
    Retrieves the enabled key-value pair of code and label, for a specific data_model.

    Returns
    ------
    Dict[str, str]
        The datasets.
    """
    data_model_code, data_model_version = data_model.split(":")

    datasets_rows = sqlite.execute_and_fetchall(
        f"""
        SELECT code, label
        FROM datasets
        WHERE data_model_id =
        (
            SELECT data_model_id
            FROM data_models
            WHERE code = '{data_model_code}'
            AND version = '{data_model_version}'
        )
        AND status = 'ENABLED'
        """
    )
    datasets = {code: label for code, label in datasets_rows}
    return datasets


@sql_injection_guard(data_model=is_datamodel)
def get_data_model_cdes(data_model: str) -> CommonDataElements:
    """
    Retrieves the cdes of the specific data_model.

    Returns
    ------
    CommonDataElements
        A CommonDataElements object
    """
    data_model_code, data_model_version = data_model.split(":")

    cdes_rows = sqlite.execute_and_fetchall(
        f"""
        SELECT code, metadata FROM "{data_model_code}:{data_model_version}_variables_metadata"
        """
    )

    cdes = CommonDataElements(
        values={
            code: CommonDataElement.parse_raw(metadata) for code, metadata in cdes_rows
        }
    )

    return cdes


@sql_injection_guard(data_model=is_datamodel)
def get_data_model_attributes(data_model: str) -> DataModelAttributes:
    """
    Retrieves the attributes, for a specific data_model.

    Returns
    ------
    DataModelAttributes
    """
    data_model_code, data_model_version = data_model.split(":")

    attributes = sqlite.execute_and_fetchall(
        f"""
        SELECT properties
        FROM data_models
        WHERE code = '{data_model_code}'
        AND version = '{data_model_version}'
        """
    )

    attributes = json.loads(attributes[0][0])
    return DataModelAttributes(
        tags=attributes["tags"], properties=attributes["properties"]
    )


def check_database_connection():
    """
    Check that the connection with the database is working.
    """
    result = sqlite.execute_and_fetchall(f"SELECT '{HEALTHCHECK_VALIDATION_STRING}'")
    assert result[0][0] == HEALTHCHECK_VALIDATION_STRING
