import json
import logging

import requests

from mipengine.tests.controller.api import algorithms_url


def test_get_algorithms():
    logging.info("---------- TEST : Get algorithms")
    request = requests.get(algorithms_url)
    result = json.loads(request.text)
    assert len(result) > 0


def test_get_demo_algorithm():
    logging.info("---------- TEST : Get demo algorithms")
    request = requests.get(algorithms_url)
    algorithms = json.loads(request.text)
    found = False
    for algorithm in algorithms:
        if algorithm["name"] == "demo":
            assert algorithm == json.loads(get_demo_algorithm())
            found = True
    assert found is True


def get_demo_algorithm() -> str:
    return """
    {
        "name": "demo",
        "crossvalidation":
        {
            "parameters":
            {
                "type":
                {
                    "multiple": false,
                    "enums": ["k_fold", "leave_dataset_out"],
                    "notblank": true,
                    "min": null,
                    "default": "k_fold",
                    "label": "Type of cross-validation",
                    "desc": "Type of cross-validation",
                    "type": "text",
                    "max": null
                },
                "nsplits":
                {
                    "multiple": false,
                    "enums": null,
                    "notblank": true,
                    "min": 2,
                    "default": 5,
                    "label": "Number of splits",
                    "desc": "Number of splits",
                    "type": "int",
                    "max": 20
                },
                "metrics":
                {
                    "multiple": true,
                    "enums": ["precision", "recall", "auc", "roc", "confusion_matrix", "f1_score"],
                    "notblank": true,
                    "min": null,
                    "default": null,
                    "label": "Metrics",
                    "desc": "Cross-validation metrics.",
                    "type": "text",
                    "max": null
                }
            },
            "label": "Cross Validation",
            "desc": "Module for performing cross validation on supervised learning models."
        },
        "inputdata":
        {
            "x":
            {
                "types": ["real", "int", "text"],
                "stattypes": ["numerical", "nominal"],
                "multiple": true,
                "enumslen": null,
                "notblank": true,
                "label": "features",
                "desc": "Features used in my algorithm."
            },
            "y":
            {
                "types": ["text", "int"],
                "stattypes": ["nominal"],
                "multiple": false,
                "enumslen": 4,
                "notblank": true,
                "label": "target",
                "desc": "Target variable for my algorithm."
            },
            "pathology":
            {
                "types": ["text"],
                "stattypes": null,
                "multiple": false,
                "enumslen": null,
                "notblank": true,
                "label": "Pathology of the data.",
                "desc": "The pathology that the algorithm will run on."
            },
            "datasets":
            {
                "types": ["text"],
                "stattypes": null,
                "multiple": true,
                "enumslen": null,
                "notblank": true,
                "label": "Set of data to use.",
                "desc": "The set of data to run the algorithm on."
            },
            "filter":
            {
                "types": ["jsonObject"],
                "stattypes": null,
                "multiple": false,
                "enumslen": null,
                "notblank": false,
                "label": "filter on the data.",
                "desc": "Features used in my algorithm."
            }
        },
        "parameters":
        {
            "my_enum_param":
            {
                "multiple": false,
                "enums": ["a", "b", "c"],
                "notblank": true,
                "min": null,
                "default": "a",
                "label": "Some param",
                "desc": "Example of parameter with enumerations.",
                "type": "text",
                "max": null
            },
            "my_int_param":
            {
                "multiple": false,
                "enums": null,
                "notblank": true,
                "min": 2,
                "default": 4,
                "label": "Some param",
                "desc": "Example of integer param.",
                "type": "int",
                "max": 4
            },
            "list_param":
            {
                "multiple": true,
                "enums": null,
                "notblank": false,
                "min": 0,
                "default": [0.8, 0.95],
                "label": "Some param",
                "desc": "Example of list of floats param.",
                "type": "real",
                "max": 1
            }
        },
        "label": "DEMO",
        "desc": "This is a demo algorithm."
    }
    """
