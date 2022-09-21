import json
import re

import pytest
import requests

from tests.prod_env_tests import algorithms_url


def get_parametrization_list_exception_cases():
    parametrization_list = []

    # ~~~~~~~~~~exception case 1~~~~~~~~~~
    algorithm_name = "logistic_regression"
    request_dict = {
        "wrong_name": {
            "data_model": "dementia:0.1",
            "datasets": ["test_dataset1", "test_dataset2"],
            "x": ["test_cde1", "test_cde2"],
            "y": ["test_cde3"],
        }
    }
    expected_response = (400, ".*Algorithm execution request malformed")
    parametrization_list.append((algorithm_name, request_dict, expected_response))

    # ~~~~~~~~~~exception case 2~~~~~~~~~~
    algorithm_name = "logistic_regression"
    request_dict = {
        "inputdata": {
            "data_model": "non_existing",
            "datasets": ["test_dataset1", "test_dataset2"],
            "x": ["test_cde1", "test_cde2"],
            "y": ["test_cde3"],
        },
    }

    expected_response = (460, "Data model .* does not exist.*")
    parametrization_list.append((algorithm_name, request_dict, expected_response))

    # ~~~~~~~~~~exception case 3~~~~~~~~~~
    algorithm_name = "logistic_regression"
    request_dict = {
        "inputdata": {
            "data_model": "dementia:0.1",
            "datasets": ["edsd0"],
            "x": ["lefthippocampus"],
            "y": ["alzheimerbroadcategory"],
            "filters": {
                "condition": "AND",
                "rules": [
                    {
                        "condition": "OR",
                        "rules": [
                            {
                                "id": "subjectage",
                                "field": "subjectage",
                                "type": "real",
                                "input": "number",
                                "operator": "greater",
                                "value": 200.0,
                            }
                        ],
                    }
                ],
                "valid": True,
            },
        },
        "parameters": {"positive_class": "AD"},
    }

    expected_response = (
        461,
        "The algorithm could not run with the input provided because there are insufficient data.",
    )
    parametrization_list.append((algorithm_name, request_dict, expected_response))

    return parametrization_list


@pytest.mark.parametrize(
    "algorithm_name, request_dict, expected_response",
    get_parametrization_list_exception_cases(),
)
def test_post_algorithm_error(algorithm_name, request_dict, expected_response):
    algorithm_url = algorithms_url + "/" + algorithm_name
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    request_json = json.dumps(request_dict)
    response = requests.post(algorithm_url, data=request_json, headers=headers)
    exp_response_status, exp_response_message = expected_response
    assert (
        response.status_code == exp_response_status
    ), f"Response message: {response.text}"
    assert re.search(exp_response_message, response.text)
