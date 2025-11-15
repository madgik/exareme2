import json

import pytest
import requests

from tests.standalone_tests.conftest import ALGORITHMS_URL
from tests.standalone_tests.conftest import COMMON_IP
from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.controller.workers_communication_helper import (
    get_celery_task_signature,
)
from tests.standalone_tests.std_output_logger import StdOutputLogger


@pytest.mark.slow
def test_processes_garbage_collect(
    globalworker_worker_service,
    localworker1_worker_service,
    controller_service_with_localworker1,
    localworker1_celery_app,
):
    start_flower_server_task_signature = get_celery_task_signature(
        "start_flower_server"
    )
    async_result = localworker1_celery_app.queue_task(
        task_signature=start_flower_server_task_signature,
        logger=StdOutputLogger(),
        request_id="test_bro",
        algorithm_folder_path="./exareme2/algorithms/flower/logistic_regression_fedaverage_flower",
        number_of_clients=1,
        server_address=f"{COMMON_IP}:8080",
        data_model="data_model:1",
        datasets=["dataset1", "dataset2"],
    )
    localworker1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    request_dict = {
        "inputdata": {
            "y": ["gender"],
            "x": ["lefthippocampus"],
            "data_model": "dementia:0.1",
            "datasets": [
                "ppmi0",
                "ppmi1",
                "ppmi2",
                "ppmi3",
            ],
            "validation_datasets": ["ppmi_test"],
            "filters": None,
        },
    }

    algorithm_url = ALGORITHMS_URL + "/" + "logistic_regression_fedaverage_flower"

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        algorithm_url,
        data=json.dumps(request_dict),
        headers=headers,
    )
    assert response.status_code == 200
