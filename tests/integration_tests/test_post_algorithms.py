import json
import re

import pytest
import requests
import numpy as np

from tests.integration_tests import algorithms_url

test_cases_post_algorithm_success = [
    (
        "logistic_regression",
        {
            "inputdata": {
                "pathology": "dementia",
                "datasets": ["edsd"],
                "x": [
                    "lefthippocampus",
                    "righthippocampus",
                    "rightppplanumpolare",
                    "leftamygdala",
                    "rightamygdala",
                ],
                "y": ["alzheimerbroadcategory"],
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
                                    "righthippocampus",
                                    "rightppplanumpolare",
                                    "leftamygdala",
                                    "rightamygdala",
                                    "alzheimerbroadcategory",
                                ]
                            ]
                        },
                    ],
                    "valid": True,
                },
            },
            "parameters": {"classes": ["AD", "CN"]},
        },
    ),
]


@pytest.mark.parametrize(
    "algorithm_name, request_body", test_cases_post_algorithm_success
)
def test_post_algorithm_success(algorithm_name, request_body):
    algorithm_url = algorithms_url + "/" + algorithm_name
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        algorithm_url, data=json.dumps(request_body), headers=headers
    )
    assert response.status_code == 200
    result = json.loads(response.text)
    expected_data = [
        ["lefthippocampus", -3.809188],
        ["righthippocampus", 4.595969],
        ["rightppplanumpolare", 3.6549711],
        ["leftamygdala", -2.4617643],
        ["rightamygdala", -11.787596],
    ]
    assert result["data"] == expected_data


test_cases_post_algorithm_failure = [
    (
        "logistic_regression",
        {
            "wrong_name": {
                "pathology": "dementia",
                "datasets": ["test_dataset1", "test_dataset2"],
                "x": ["test_cde1", "test_cde2"],
                "y": ["test_cde3"],
            },
        },
        (400, ".* proper format."),
    ),
    (
        "logistic_regression",
        {
            "inputdata": {
                "pathology": "non_existing",
                "datasets": ["test_dataset1", "test_dataset2"],
                "x": ["test_cde1", "test_cde2"],
                "y": ["test_cde3"],
            },
        },
        (200, "Pathology .* does not exist.*"),
    ),
]


@pytest.mark.parametrize(
    "algorithm_name, request_body, exp_response", test_cases_post_algorithm_failure
)
def test_post_algorithm_error(algorithm_name, request_body, exp_response):
    algorithm_url = algorithms_url + "/" + algorithm_name
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        algorithm_url, data=json.dumps(request_body), headers=headers
    )
    exp_response_status, exp_response_message = exp_response
    assert response.status_code == exp_response_status
    assert re.search(exp_response_message, response.text)
