import uuid

import pytest

from exareme2.worker_communication import DatasetsInfoPerDataModel
from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.controller.workers_communication_helper import (
    get_celery_task_signature,
)
from tests.standalone_tests.std_output_logger import StdOutputLogger

label_identifier = "test_get_datasets_per_data_model"


def setup_data_table_in_db(datasets_per_data_model, cursor):
    cursor.run("DELETE FROM datasets;")
    cursor.run("DELETE FROM data_models;")
    data_model_id = 0
    dataset_id = 0
    for data_model in datasets_per_data_model.keys():
        data_model_id += 1
        dataset_id += 1
        data_model_code, data_model_version = data_model.split(":")
        sql_query = f"""INSERT INTO "data_models" (code, version, label, status, properties) VALUES ('{data_model_code}', '{data_model_version}', '{label_identifier}', 'ENABLED', null);"""
        cursor.run(sql_query)
        for dataset_name in datasets_per_data_model[data_model]:
            dataset_id += 1
            sql_query = f"""INSERT INTO "datasets" (data_model_id, code, label, csv_path, status, properties) VALUES ({data_model_id}, '{dataset_name}', '{label_identifier}', 'csv_path', 'ENABLED', null);"""
            cursor.run(sql_query)


data_model1 = "data_model1:0.1"
data_model2 = "data_model2:0.1"
data_model3 = "data_model3:0.1"
test_cases_get_worker_info_datasets = [
    {
        data_model1: [
            "dataset1",
            "dataset2",
            "dataset5",
            "dataset7",
            "dataset15",
        ],
        data_model2: ["dataset3"],
        data_model3: ["dataset4"],
    },
    {
        data_model1: [
            "dataset123",
        ],
        data_model2: [
            "dataset123",
        ],
    },
    {
        data_model1: [],
    },
    {},
]


@pytest.mark.slow
@pytest.mark.very_slow
@pytest.mark.parametrize(
    "expected_datasets_per_data_model",
    test_cases_get_worker_info_datasets,
)
def test_get_worker_datasets_per_data_model(
    expected_datasets_per_data_model,
    globalworker_worker_service,
    globalworker_celery_app,
    use_globalworker_database,
    globalworker_sqlite_db_cursor,
    init_data_globalworker,
):
    request_id = "test_worker_info_" + uuid.uuid4().hex + "_request"
    setup_data_table_in_db(
        expected_datasets_per_data_model, globalworker_sqlite_db_cursor
    )
    task_signature = get_celery_task_signature("get_worker_datasets_per_data_model")
    async_result = globalworker_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
    )

    datasets_per_data_model = globalworker_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )
    dataset_infos_per_data_model = DatasetsInfoPerDataModel.parse_raw(
        datasets_per_data_model
    )

    expected_data_models = set(expected_datasets_per_data_model.keys())
    available_data_models = set(
        dataset_infos_per_data_model.datasets_info_per_data_model.keys()
    )

    assert expected_data_models.issubset(available_data_models)

    for (
        data_model,
        datasets,
    ) in expected_datasets_per_data_model.items():
        expected_datasets = set(datasets)

        available_datasets = set(
            dataset_info.code
            for dataset_info in dataset_infos_per_data_model.datasets_info_per_data_model[
                data_model
            ]
        )
        assert expected_datasets.issubset(available_datasets)
