import json

import requests

from tests.prod_env_tests import algorithms_url


def test_get_algorithms():
    request = requests.get(algorithms_url)
    result = json.loads(request.text)
    assert len(result) > 0


def test_get_logistic_regression():
    request = requests.get(algorithms_url)
    algorithms = json.loads(request.text)
    algorithm_names = [algorithm["name"] for algorithm in algorithms]
    assert "logistic_regression" in algorithm_names


def test_logistic_regression_has_longitudinal_transformer():
    request = requests.get(algorithms_url)
    result = json.loads(request.text)

    for algorithm in result:
        if algorithm["name"] == "logistic_regression":
            if not algorithm["preprocessing"]:
                pytest.fail(
                    "Logistic Regression should have the 'longitudinal_transform' as preprocessing step."
                )
            for transformer in algorithm["preprocessing"]:
                if transformer["name"] == "longitudinal_transformer":
                    break
            else:
                pytest.fail(
                    "Logistic Regression should have the 'longitudinal_transform' as preprocessing step."
                )
            break
    else:
        pytest.fail("Logistic Regression algorithm was not found")


def test_descriptive_statistics_has_no_longitudinal_transformer():
    request = requests.get(algorithms_url)
    result = json.loads(request.text)

    for algorithm in result:
        if algorithm["name"] == "descriptive_stats":
            if algorithm["preprocessing"]:
                pytest.fail(
                    "Descriptive Statistics should NOT have the 'longitudinal_transform' as preprocessing step."
                )
            break
    else:
        pytest.fail("Descriptive Statistics algorithm was not found.")
