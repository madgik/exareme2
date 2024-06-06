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
    localworker1_worker_service,
    load_data_localworker1,
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
        algorithm_name="logistic_regression",
        number_of_clients=1,
        server_address=f"{COMMON_IP}:8080",
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
            "filters": None,
        },
        "type": "flower",
    }

    algorithm_url = ALGORITHMS_URL + "/" + "logistic_regression"

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        algorithm_url,
        data=json.dumps(request_dict),
        headers=headers,
    )
    assert response.status_code == 200
