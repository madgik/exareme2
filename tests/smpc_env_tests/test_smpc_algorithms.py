import json
import re

import numpy as np

import pytest
import requests

from tests.prod_env_tests import algorithms_url


def get_parametrization_list_success_cases():
    parametrization_list = []

    # ~~~~~~~~~~success case 1~~~~~~~~~~
    algorithm_name = "smpc_standard_deviation"
    request_dict = {
        "inputdata": {
            "data_model": "dementia",
            "data_model_version": "0.1",
            "datasets": ["edsd"],
            "x": [
                "lefthippocampus",
            ],
            "filters": {
                "condition": "AND",
                "rules": [
                    {
                        "id": "dataset",
                        "type": "string",
                        "value": ["edsd"],
                        "operator": "in",
                    },
                    {
                        "condition": "AND",
                        "rules": [
                            {
                                "id": variable,
                                "type": "string",
                                "operator": "is_not_null",
                                "value": None,
                            }
                            for variable in [
                                "lefthippocampus",
                            ]
                        ],
                    },
                ],
                "valid": True,
            },
        },
    }
    expected_response = {
        "title": "Standard Deviation",
        "columns": [
            {"name": "variable", "data": ["lefthippocampus"], "type": "STR"},
            {"name": "std_deviation", "data": [0.3611575592573076], "type": "FLOAT"},
        ],
    }
    parametrization_list.append((algorithm_name, request_dict, expected_response))
    # END ~~~~~~~~~~success case 1~~~~~~~~~~

    # ~~~~~~~~~~success case 2~~~~~~~~~~
    algorithm_name = "smpc_standard_deviation"
    request_dict = {
        "inputdata": {
            "data_model": "dementia",
            "data_model_version": "0.1",
            "datasets": ["edsd"],
            "x": [
                "lefthippocampus",
            ],
            "filters": {
                "condition": "AND",
                "rules": [
                    {
                        "id": "dataset",
                        "type": "string",
                        "value": ["edsd"],
                        "operator": "in",
                    },
                    {
                        "condition": "AND",
                        "rules": [
                            {
                                "id": variable,
                                "type": "string",
                                "operator": "is_not_null",
                                "value": None,
                            }
                            for variable in [
                                "lefthippocampus",
                            ]
                        ],
                    },
                ],
                "valid": True,
            },
        },
        "flags": {
            "smpc": True,
        },
    }
    expected_response = {
        "title": "Standard Deviation",
        "columns": [
            {"name": "variable", "data": ["lefthippocampus"], "type": "STR"},
            {"name": "std_deviation", "data": [0.3611575592573076], "type": "FLOAT"},
        ],
    }
    parametrization_list.append((algorithm_name, request_dict, expected_response))
    # END ~~~~~~~~~~success case 2~~~~~~~~~~
    return parametrization_list


# @pytest.mark.skip(
#     reason="SMPC is not deployed in the CI yet. https://team-1617704806227.atlassian.net/browse/MIP-344"
# )
@pytest.mark.parametrize(
    "algorithm_name, request_dict, expected_response",
    get_parametrization_list_success_cases(),
)
def test_post_smpc_algorithm(algorithm_name, request_dict, expected_response):
    algorithm_url = algorithms_url + "/" + algorithm_name

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        algorithm_url,
        data=json.dumps(request_dict),
        headers=headers,
    )
    assert response.status_code == 200
    assert json.loads(response.text) == expected_response


def get_parametrization_list_exception_cases():
    parametrization_list = []
    algorithm_name = "smpc_standard_deviation"
    request_dict = {
        "inputdata": {
            "data_model": "dementia",
            "data_model_version": "0.1",
            "datasets": ["edsd"],
            "x": [
                "lefthippocampus",
            ],
            "filters": {
                "condition": "AND",
                "rules": [
                    {
                        "id": "dataset",
                        "type": "string",
                        "value": ["edsd"],
                        "operator": "in",
                    },
                    {
                        "condition": "AND",
                        "rules": [
                            {
                                "id": variable,
                                "type": "string",
                                "operator": "is_not_null",
                                "value": None,
                            }
                            for variable in [
                                "lefthippocampus",
                            ]
                        ],
                    },
                ],
                "valid": True,
            },
        },
        "flags": {
            "smpc": False,
        },
    }

    expected_response = (
        462,
        "The computation cannot be made without SMPC.",
    )

    parametrization_list.append((algorithm_name, request_dict, expected_response))

    return parametrization_list


# @pytest.mark.skip(
#     reason="SMPC is not deployed in the CI yet. https://team-1617704806227.atlassian.net/browse/MIP-344"
# )
@pytest.mark.parametrize(
    "algorithm_name, request_dict, expected_response",
    get_parametrization_list_exception_cases(),
)
def test_post_smpc_algorithm_exception(algorithm_name, request_dict, expected_response):
    algorithm_url = algorithms_url + "/" + algorithm_name

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        algorithm_url,
        data=json.dumps(request_dict),
        headers=headers,
    )
    exp_response_status, exp_response_message = expected_response
    assert response.status_code == exp_response_status
    assert re.search(exp_response_message, response.text)
