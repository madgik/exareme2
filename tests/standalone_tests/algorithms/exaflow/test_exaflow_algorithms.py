import json

import pytest
import requests

from tests.standalone_tests.conftest import ALGORITHMS_URL


def get_algorithms():
    return [
        pytest.param(
            "compute_average",
            {
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
            },
            id="Dummy avg algorithm",
        )
    ]


@pytest.mark.slow
@pytest.mark.parametrize(
    "algorithm_name,request_dict",
    get_algorithms(),
)
def test_exaflow_algorithms(
    algorithm_name,
    request_dict,
    localworker1_worker_service,
    localworker2_worker_service,
    load_data_localworker1,
    load_data_localworker2,
    globalworker_worker_service,
    init_data_globalworker,
    controller_service_with_localworker_1_2,
):
    algorithm_url = ALGORITHMS_URL + "/" + algorithm_name
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        algorithm_url,
        data=json.dumps(request_dict),
        headers=headers,
    )
    assert response.status_code == 200, f"Response message: {response.text}"
