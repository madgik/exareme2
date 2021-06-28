import json
import logging
import re
import threading

import pytest
import requests

from tests.integration_tests import algorithms_url

test_cases_post_algorithm_success = [
    (
        "logistic_regression",
        {
            "inputdata": {
                "pathology": "dementia",
                "datasets": ["demo_data"],
                "filters": {
                    "valid": True,
                    "condition": "AND",
                    "rules": [
                        {
                            "id": "alzheimerbroadcategory_bin",
                            "type": "column",
                            "value": None,
                            "operator": "is_not_null",
                        },
                        {
                            "id": "dataset",
                            "type": "string",
                            "value": ["demo_data"],
                            "operator": "in",
                        },
                        {
                            "id": "lefthippocampus",
                            "type": "column",
                            "value": None,
                            "operator": "is_not_null",
                        },
                        {
                            "id": "righthippocampus",
                            "type": "column",
                            "value": None,
                            "operator": "is_not_null",
                        },
                    ],
                },
                "x": ["lefthippocampus", "righthippocampus"],
                "y": ["alzheimerbroadcategory_bin"],
            },
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
