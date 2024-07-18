import json
from typing import Dict
from typing import List

from exareme2.worker import config as worker_config
from exareme2.worker.exareme2.monetdb.guard import is_datamodel
from exareme2.worker.exareme2.monetdb.guard import sql_injection_guard
from exareme2.worker.worker_info import sqlite
from exareme2.worker_communication import CommonDataElement
from exareme2.worker_communication import CommonDataElements
from exareme2.worker_communication import DataModelAttributes
from exareme2.worker_communication import DatasetInfo

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


def convert_absolute_dataset_path_to_relative(dataset_path: str) -> str:
    return dataset_path.split(str(worker_config.data_path))[-1]


@sql_injection_guard(data_model=is_datamodel)
def get_dataset_infos(data_model: str) -> List[DatasetInfo]:
    """
    Retrieves the enabled dataset, for a specific data_model.

    Returns
    ------
    Dict[str, str]
        The datasets.
    """
    data_model_code, data_model_version = data_model.split(":")

    datasets_rows = sqlite.execute_and_fetchall(
        f"""
        SELECT code, label, csv_path
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
    return [
        DatasetInfo(
            code=row[0],
            label=row[1],
            csv_path=convert_absolute_dataset_path_to_relative(row[2])
            if row[2] is not None
            else None,
        )
        for row in datasets_rows
    ]


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
