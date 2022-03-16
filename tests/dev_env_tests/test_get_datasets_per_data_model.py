import json
import subprocess
import uuid

import fasteners
import pytest

from tests.dev_env_tests.nodes_communication import get_celery_app
from tests.dev_env_tests.nodes_communication import get_celery_task_signature

label_identifier = "test_get_datasets_per_data_model"


@pytest.fixture(autouse=True)
def run_tests_sequentially():
    with fasteners.InterProcessLock("semaphore.lock"):
        yield


def setup_data_table_in_db(node_id, datasets_per_data_model):
    data_model_id = 2
    dataset_id = 4
    for data_model in datasets_per_data_model.keys():
        data_model_id += 1
        dataset_id += 1
        data_model_code, data_model_version = data_model.split(":")
        sql_query = f"""INSERT INTO "mipdb_metadata"."data_models" VALUES ({data_model_id}, '{data_model_code}', '{data_model_version}', '{label_identifier}', 'ENABLED', null);"""
        cmd = f'docker exec -i monetdb-{node_id} mclient db -s "{sql_query}"'
        subprocess.call(cmd, shell=True)
        for dataset_name in datasets_per_data_model[data_model]:
            dataset_id += 1
            sql_query = f"""INSERT INTO "mipdb_metadata"."datasets" VALUES ({dataset_id}, {data_model_id}, '{dataset_name}', '{label_identifier}', 'ENABLED', null);"""
            cmd = f'docker exec -i monetdb-{node_id} mclient db -s "{sql_query}"'
            subprocess.call(cmd, shell=True)


# The cleanup task cannot be used because it requires specific table name convention
# that doesn't fit with the initial data table names
def teardown_data_tables_in_db(node_id):
    sql_query = (
        f"DELETE FROM mipdb_metadata.datasets WHERE label = '{label_identifier}';"
    )
    cmd = f'docker exec -i monetdb-{node_id} mclient db -s "{sql_query}" '
    subprocess.call(cmd, shell=True)
    sql_query = (
        f"DELETE FROM mipdb_metadata.data_models WHERE label = '{label_identifier}';"
    )
    cmd = f'docker exec -i monetdb-{node_id} mclient db -s "{sql_query}" '
    subprocess.call(cmd, shell=True)


data_model1 = "data_model1:0.1"
data_model2 = "data_model2:0.1"
data_model3 = "data_model3:0.1"
test_cases_get_node_info_datasets = [
    (
        "localnode1",
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
    ),
    (
        "localnode1",
        {
            data_model1: [
                "dataset123",
            ],
            data_model2: [
                "dataset123",
            ],
        },
    ),
    (
        "localnode1",
        {
            data_model1: [],
        },
    ),
    (
        "localnode1",
        {},
    ),
]


@pytest.mark.parametrize(
    "node_id, expected_datasets_per_data_model",
    test_cases_get_node_info_datasets,
)
def test_get_node_datasets_per_data_model(node_id, expected_datasets_per_data_model):
    teardown_data_tables_in_db(node_id)
    request_id = "test_node_info_" + uuid.uuid4().hex + "_request"
    setup_data_table_in_db(node_id, expected_datasets_per_data_model)
    node_app = get_celery_app(node_id)
    get_node_datasets_per_data_model_signature = get_celery_task_signature(
        node_app, "get_node_datasets_per_data_model"
    )
    datasets_per_data_model = get_node_datasets_per_data_model_signature.delay(
        request_id=request_id
    ).get()
    expected_datasets_per_data_model["dementia:0.1"] = [
        "ppmi",
        "edsd",
        "desd-synthdata",
    ]
    expected_datasets_per_data_model["tbi:0.1"] = ["dummy_tbi"]
    print(set(datasets_per_data_model.keys()))
    print(set(expected_datasets_per_data_model.keys()))
    assert set(datasets_per_data_model.keys()) == set(
        expected_datasets_per_data_model.keys()
    )
    for data_model in expected_datasets_per_data_model.keys():
        assert set(datasets_per_data_model[data_model]) == set(
            expected_datasets_per_data_model[data_model]
        )

    # teardown_data_tables_in_db(node_id)
