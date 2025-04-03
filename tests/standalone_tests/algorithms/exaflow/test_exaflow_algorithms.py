import json

import requests

from tests.standalone_tests.conftest import ALGORITHMS_URL


def test_exaflow_dummy_algorithm(
    localworker1_worker_service,
    load_data_localworker1,
    globalworker_worker_service,
    controller_service_with_localworker1,
):
    algorithm_url = ALGORITHMS_URL + "/" + "compute_average"
    request_dict = {
        "inputdata": {
            "data_model": "dementia:0.1",
            "datasets": ["edsd0"],
            "y": [
                "lefthippocampus",
            ],
            "filters": {
                "condition": "AND",
                "rules": [
                    {
                        "id": variable,
                        "type": "string",
                        "operator": "is_not_null",
                        "value": None,
                    }
                    for variable in {"lefthippocampus"}
                ],
            },
        },
        "type": "exaflow",
    }
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        algorithm_url,
        data=json.dumps(request_dict),
        headers=headers,
    )
    assert response.status_code == 200, f"Response message: {response.text}"
