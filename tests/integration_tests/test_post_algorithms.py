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
                "datasets": ["ppmi"],
                "x": [
                    "lefthippocampus",
                    "righthippocampus",
                    "rightppplanumpolare",
                    "leftamygdala",
                    "rightamygdala",
                ],
                "y": ["parkinsonbroadcategory"],
            },
            "parameters": {"classes": ["PD", "CN"]},
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
    result = [coeff for _, _, coeff in json.loads(response.text)]
    expected = np.array([0.864517, 0.3577170, 0.475236, -2.682983, -2.615825])
    assert np.isclose(result, expected).all()


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
