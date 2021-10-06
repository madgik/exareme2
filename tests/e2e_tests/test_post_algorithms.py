import json
import re

import pytest
import requests
import numpy as np

from tests.e2e_tests import algorithms_url

from mipengine.controller.api.algorithm_request_dto import (
    AlgorithmInputDataDTO,
    AlgorithmRequestDTO,
)


def get_parametrization_list_success_cases():

    parametrization_list = []

    inputdata_dto = AlgorithmInputDataDTO(
        pathology="dementia",
        datasets=["edsd"],
        x=[
            "lefthippocampus",
            "righthippocampus",
            "rightppplanumpolare",
            "leftamygdala",
            "rightamygdala",
        ],
        y=["alzheimerbroadcategory"],
        filters={
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
                    ],
                },
            ],
            "valid": True,
        },
    )
    algorithm_request_dto = AlgorithmRequestDTO(
        inputdata=inputdata_dto,
        parameters={"classes": ["AD", "CN"]},
    )

    algorithm_name = "logistic_regression"

    expected_response = {
        "title": "Logistic Regression Coefficients",
        "columns": [
            {"name": "variable", "type": "string"},
            {"name": "coefficient", "type": "number"},
        ],
        "data": [
            ["lefthippocampus", "-3.809188"],
            ["righthippocampus", "4.595969"],
            ["rightppplanumpolare", "3.6549711"],
            ["leftamygdala", "-2.4617643"],
            ["rightamygdala", "-11.787596"],
        ],
    }

    parametrization_list.append(
        (algorithm_name, algorithm_request_dto, expected_response)
    )

    return parametrization_list


@pytest.mark.parametrize(
    "algorithm_name, request_dto, expected_response",
    get_parametrization_list_success_cases(),
)
def test_post_algorithm_success(algorithm_name, request_dto, expected_response):
    algorithm_url = algorithms_url + "/" + algorithm_name

    request_json = request_dto.json()

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        algorithm_url,
        data=request_json,
        headers=headers,
    )
    assert response.status_code == 200

    assert response.json() == expected_response


def get_parametrization_list_exception_cases():
    parametrization_list = [
        (
            "logistic_regression",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    pathology="non_existing",
                    datasets=["test_dataset1", "test_dataset2"],
                    x=["test_cde1", "test_cde2"],
                    y=["test_cde3"],
                )
            ),
            (460, "Pathology .* does not exist.*"),
        ),
    ]
    return parametrization_list


@pytest.mark.parametrize(
    "algorithm_name, request_dto, exp_response",
    get_parametrization_list_exception_cases(),
)
def test_post_algorithm_error(algorithm_name, request_dto, exp_response):
    algorithm_url = algorithms_url + "/" + algorithm_name
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    request_json = request_dto.json()
    response = requests.post(algorithm_url, data=request_json, headers=headers)
    exp_response_status, exp_response_message = exp_response
    assert response.status_code == exp_response_status
    assert re.search(exp_response_message, response.text)
