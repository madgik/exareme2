import json

import pytest
import requests

from tests.prod_env_tests import algorithms_url


def get_parametrization_list_success_cases():
    algorithm_name = "standard_deviation_pos_and_kw_args"
    request_dict = {
        "inputdata": {
            "pathology": "dementia",
            "datasets": ["edsd"],
            "x": [
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
    }

    parametrization_list = []
    parametrization_list.append((algorithm_name, request_dict))

    return parametrization_list


@pytest.mark.parametrize(
    "algorithm_name, request_dict",
    get_parametrization_list_success_cases(),
)
def test_pos_and_kw_args_in_algorithm_flow(algorithm_name, request_dict):
    algorithm_url = algorithms_url + "/" + algorithm_name

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        algorithm_url,
        data=json.dumps(request_dict),
        headers=headers,
    )
    assert response.status_code == 200
