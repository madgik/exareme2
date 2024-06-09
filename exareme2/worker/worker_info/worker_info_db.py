import json
from typing import Dict
from typing import List

from exareme2.worker.exareme2.monetdb.guard import is_datamodel
from exareme2.worker.exareme2.monetdb.guard import sql_injection_guard
from exareme2.worker.exareme2.monetdb.monetdb_facade import db_execute_and_fetchall
from exareme2.worker_communication import DataModelMetadata
from exareme2.worker_communication import parse_data_model_metadata

HEALTHCHECK_VALIDATION_STRING = "HEALTHCHECK"


def get_data_models() -> List[str]:
    """
    Retrieves the enabled data_models from the database.

    Returns
    ------
    List[str]
        The data_models.
    """

    data_models_code_and_version = db_execute_and_fetchall(
        f"""SELECT code, version
            FROM "mipdb_metadata"."data_models"
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

    datasets_rows = db_execute_and_fetchall(
        f"""
        SELECT code, label
        FROM "mipdb_metadata"."datasets"
        WHERE data_model_id =
        (
            SELECT data_model_id
            FROM "mipdb_metadata"."data_models"
            WHERE code = '{data_model_code}'
            AND version = '{data_model_version}'
        )
        AND status = 'ENABLED'
        """
    )
    datasets = {code: label for code, label in datasets_rows}
    return datasets


def get_data_model_metadata() -> Dict[str, str]:
    """
    Retrieves the metadata for each data_model.

    Returns
    ------
    Dict[str, DataModelMetadata]
    """
    result = db_execute_and_fetchall(
        f"""
        SELECT code, version, properties
        FROM "mipdb_metadata"."data_models"
        """
    )
    return {
        f"{code}:{version}": json.loads(properties)["properties"]["cdes"]
        for code, version, properties in result
    }


def get_datasets_per_data_model() -> Dict[str, List[str]]:
    result = db_execute_and_fetchall(
        f"""SELECT data_model_id, code, version
                FROM "mipdb_metadata"."data_models"
                WHERE status = 'ENABLED'
            """
    )
    data_models = {
        data_model_id: code + ":" + version for data_model_id, code, version in result
    }
    result = db_execute_and_fetchall(
        f"""SELECT data_model_id, code
            FROM "mipdb_metadata"."datasets"
            WHERE status = 'ENABLED'
        """
    )
    datasets_per_data_model = {}
    for data_model_id, code in result:
        data_model = data_models[data_model_id]
        if data_model in datasets_per_data_model:
            datasets_per_data_model[data_model].append(code)
        else:
            datasets_per_data_model[data_model] = [code]
    return datasets_per_data_model


def check_database_connection():
    """
    Check that the connection with the database is working.
    """
    result = db_execute_and_fetchall(f"SELECT '{HEALTHCHECK_VALIDATION_STRING}'")
    assert result[0][0] == HEALTHCHECK_VALIDATION_STRING
