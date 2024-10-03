import concurrent.futures
import json

import pytest
import requests

from tests.standalone_tests.conftest import ALGORITHMS_URL


@pytest.mark.slow
def test_parallel_requests_in_algorithm_flow(
    globalworker_worker_service,
    localworker1_worker_service,
    load_data_localworker1,
    load_test_data_globalworker,
    controller_service_with_localworker1,
):
    algorithm_name = "logistic_regression"
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
        "type": "flower",
    }

    algorithm_url = ALGORITHMS_URL + "/" + algorithm_name

    headers = {"Content-type": "application/json", "Accept": "text/plain"}

    num_requests = 5  # Number of parallel requests

    def send_request():
        response = requests.post(
            algorithm_url,
            data=json.dumps(request_dict),
            headers=headers,
        )
        assert response.status_code == 200
        return response.json()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(send_request) for _ in range(num_requests)]
        results = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]

    expected_result = {"accuracy": 0.63}

    for result in results:
        assert result == expected_result
