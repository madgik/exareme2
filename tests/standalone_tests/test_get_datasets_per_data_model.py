import subprocess
import uuid

import fasteners
import pytest

from tests.standalone_tests.conftest import MONETDB_GLOBALNODE_NAME
from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.nodes_communication_helper import get_celery_task_signature
from tests.standalone_tests.std_output_logger import StdOutputLogger

label_identifier = "test_get_datasets_per_data_model"


@pytest.fixture(autouse=True)
def run_tests_sequentially():
    with fasteners.InterProcessLock("semaphore.lock"):
        yield


def setup_data_table_in_db(datasets_per_data_model):
    data_model_id = 0
    dataset_id = 0
    for data_model in datasets_per_data_model.keys():
        data_model_id += 1
        dataset_id += 1
        data_model_code, data_model_version = data_model.split(":")
        sql_query = f"""INSERT INTO "mipdb_metadata"."data_models" VALUES ({data_model_id}, '{data_model_code}', '{data_model_version}', '{label_identifier}', 'ENABLED', null);"""
        cmd = f'docker exec -i {MONETDB_GLOBALNODE_NAME} mclient db -s "{sql_query}"'
        subprocess.call(cmd, shell=True)
        for dataset_name in datasets_per_data_model[data_model]:
            dataset_id += 1
            sql_query = f"""INSERT INTO "mipdb_metadata"."datasets" VALUES ({dataset_id}, {data_model_id}, '{dataset_name}', '{label_identifier}', 'ENABLED', null);"""
            cmd = (
                f'docker exec -i {MONETDB_GLOBALNODE_NAME} mclient db -s "{sql_query}"'
            )
            subprocess.call(cmd, shell=True)


# The cleanup task cannot be used because it requires specific table name convention
# that doesn't fit with the initial data table names
def teardown_data_tables_in_db():
    sql_query = (
        f"DELETE FROM mipdb_metadata.datasets WHERE label = '{label_identifier}';"
    )
    cmd = f'docker exec -i {MONETDB_GLOBALNODE_NAME} mclient db -s "{sql_query}" '
    subprocess.call(cmd, shell=True)
    sql_query = (
        f"DELETE FROM mipdb_metadata.data_models WHERE label = '{label_identifier}';"
    )
    cmd = f'docker exec -i {MONETDB_GLOBALNODE_NAME} mclient db -s "{sql_query}" '
    subprocess.call(cmd, shell=True)


data_model1 = "data_model1:0.1"
data_model2 = "data_model2:0.1"
data_model3 = "data_model3:0.1"
test_cases_get_node_info_datasets = [
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


@pytest.mark.parametrize(
    "expected_datasets_per_data_model",
    test_cases_get_node_info_datasets,
)
def test_get_node_datasets_per_data_model(
    expected_datasets_per_data_model,
    globalnode_node_service,
    globalnode_celery_app,
    use_globalnode_database,
    init_data_globalnode,
):
    request_id = "test_node_info_" + uuid.uuid4().hex + "_request"
    setup_data_table_in_db(expected_datasets_per_data_model)
    task_signature = get_celery_task_signature("get_node_datasets_per_data_model")
    async_result = globalnode_celery_app.queue_task(
        task_signature=task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
    )

    datasets_per_data_model = globalnode_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    assert set(datasets_per_data_model.keys()) == set(
        expected_datasets_per_data_model.keys()
    )
    for data_model in expected_datasets_per_data_model.keys():
        assert set(datasets_per_data_model[data_model]) == set(
            expected_datasets_per_data_model[data_model]
        )

    teardown_data_tables_in_db()
