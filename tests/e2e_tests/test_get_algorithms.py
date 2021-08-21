import json

import requests

from tests.e2e_tests import algorithms_url


def test_get_algorithms():
    request = requests.get(algorithms_url)
    result = json.loads(request.text)
    assert len(result) > 0


def test_get_logistic_regression():
    request = requests.get(algorithms_url)
    algorithms = json.loads(request.text)
    algorithm_names = [algorithm["name"] for algorithm in algorithms]
    assert "logistic_regression" in algorithm_names
