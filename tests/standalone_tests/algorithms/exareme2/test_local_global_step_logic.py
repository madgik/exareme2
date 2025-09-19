import json

import pytest
import requests

from tests.standalone_tests.conftest import ALGORITHMS_URL


def get_parametrization_list_success_cases():
    algorithm_name = "standard_deviation"
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
        "type": "exareme2",
    }
    return [(algorithm_name, request_dict)]


@pytest.mark.slow
@pytest.mark.parametrize(
    "algorithm_name, request_dict",
    get_parametrization_list_success_cases(),
)
def test_local_global_step_algorithms(
    algorithm_name,
    request_dict,
    monetdb_localworker1,
    localworker1_worker_service,
    load_data_localworker1_with_monetdb,
    monetdb_globalworker,
    globalworker_worker_service,
    controller_service_with_localworker1,
):
    algorithm_url = ALGORITHMS_URL + "/" + algorithm_name

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        algorithm_url,
        data=json.dumps(request_dict),
        headers=headers,
    )
    assert response.status_code == 200, f"Response message: {response.text}"
