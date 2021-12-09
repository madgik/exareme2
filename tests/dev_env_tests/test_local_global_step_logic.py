import json

import pytest
import requests

from tests.prod_env_tests import algorithms_url


def get_parametrization_list_success_cases():
    algorithm_name = "standard_deviation"
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

    expected_response = {
        "title": "Standard Deviation",
        "columns": [
            {"name": "variable", "type": "string"},
            {"name": "std_deviation", "type": "number"},
        ],
        "data": [
            ["lefthippocampus", "0.3773485998788057"],
        ],
    }
    # TODO quotes should be removed from the lefthippocampus value.
    # BUG related https://team-1617704806227.atlassian.net/browse/MIP-260

    parametrization_list = []
    parametrization_list.append((algorithm_name, request_dict, expected_response))

    return parametrization_list


@pytest.mark.parametrize(
    "algorithm_name, request_dict, expected_response",
    get_parametrization_list_success_cases(),
)
def test_local_global_step_algorithms(algorithm_name, request_dict, expected_response):
    algorithm_url = algorithms_url + "/" + algorithm_name

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        algorithm_url,
        data=json.dumps(request_dict),
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["title"] == expected_response["title"]
    assert response.json()["columns"] == expected_response["columns"]
    assert response.json()["data"][0][0] == expected_response["data"][0][0]
    # Assert the standard deviation with 4 decimals only
    assert round(float(response.json()["data"][0][1]), 4) == round(
        float(expected_response["data"][0][1]), 4
    )
