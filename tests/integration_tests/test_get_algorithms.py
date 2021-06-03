import json
import logging

import requests

from tests.integration_tests import algorithms_url


def test_get_algorithms():
    logging.info("---------- TEST : Get algorithms")
    request = requests.get(algorithms_url)
    result = json.loads(request.text)
    assert len(result) > 0


def test_get_logistic_regression():
    request = requests.get(algorithms_url)
    algorithms = json.loads(request.text)

    found = False
    for algorithm in algorithms:
        if algorithm["name"] == "logistic_regression":
            assert algorithm == json.loads(logistic_regression_json)
            found = True
    assert found is True


logistic_regression_json = """
{
    "label": "Logistic Regression",
    "inputdata":
    {
        "filter":
        {
            "label": "filter on the data.",
            "enumslen": null,
            "stattypes": null,
            "multiple": false,
            "desc": "Features used in my algorithm.",
            "notblank": false,
            "types": ["jsonObject"]
        },
        "x":
        {
            "label": "features",
            "enumslen": null,
            "stattypes": ["numerical"],
            "multiple": true,
            "desc": "Features",
            "notblank": true,
            "types": ["real"]
        },
        "pathology":
        {
            "label": "Pathology of the data.",
            "enumslen": null,
            "stattypes": null,
            "multiple": false,
            "desc": "The pathology that the algorithm will run on.",
            "notblank": true,
            "types": ["text"]
        },
        "y":
        {
            "label": "target",
            "enumslen": 2,
            "stattypes": ["nominal"],
            "multiple": false,
            "desc": "Target variable",
            "notblank": true,
            "types": ["int"]
        },
        "datasets":
        {
            "label": "Set of data to use.",
            "enumslen": null,
            "stattypes": null,
            "multiple": true,
            "desc": "The set of data to run the algorithm on.",
            "notblank": true,
            "types": ["text"]
        }
    },
    "parameters": {},
    "desc": "Logistic Regression",
    "name": "logistic_regression"
}
"""
