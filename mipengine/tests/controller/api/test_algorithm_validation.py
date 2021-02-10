import json
import logging

import requests

from mipengine.tests.controller.api import algorithms_url

demo_algorithm_path = "../../../algorithms/demo.json"


def test_wrong_algorithm_name():
    logging.info("---------- TEST : POST non existing algorithm")

    request_url = algorithms_url + "/not_existing"

    request = requests.post(request_url)

    assert request.status_code == 400

    assert request.text == "Algorithm 'not_existing' does not exist."


def test_wrong_pathology():
    logging.info("---------- TEST : POST non existing pathology")

    request_url = algorithms_url + "/demo"
    request_body = {
        "inputdata": {
            "pathology": "wrong",
            "dataset": ["adni"],
            "filter": {},
            "x": ["lefthippocampus", "righthippocampus"],
            "y": ["alzheimerbroadcategory"]
        },
        "parameters": {
            "my_enum_param": "a",
            "my_int_param": 3,
            "list_param": [0.8, 0.95]
        },
        "crossvalidation": {
            "type": "k_fold",
            "nsplits": 10,
            "metrics": ["precision", "confusion_matrix"]
        }
    }
    headers = {"Content-type": "application/json", "Accept": "text/plain"}

    request = requests.post(request_url, data=json.dumps(request_body), headers=headers)

    assert request.status_code == 200

    assert json.loads(request.text) == get_user_error_response("Pathology 'wrong' does not exist.")


def test_wrong_dataset():
    logging.info("---------- TEST : POST non existing dataset")

    request_url = algorithms_url + "/demo"
    request_body = {
        "inputdata": {
            "pathology": "dementia",
            "dataset": ["wrong"],
            "filter": {},
            "x": ["lefthippocampus", "righthippocampus"],
            "y": ["alzheimerbroadcategory"]
        },
        "parameters": {
            "my_enum_param": "a",
            "my_int_param": 3,
            "list_param": [0.8, 0.95]
        },
        "crossvalidation": {
            "type": "k_fold",
            "nsplits": 10,
            "metrics": ["precision", "confusion_matrix"]
        }
    }
    headers = {"Content-type": "application/json", "Accept": "text/plain"}

    request = requests.post(request_url, data=json.dumps(request_body), headers=headers)

    assert request.status_code == 200

    assert json.loads(request.text) == \
           get_user_error_response("Datasets '['wrong']' do not belong in pathology 'dementia'.")


def test_proper_algorithm():
    logging.info("---------- TEST : POST proper algorithm")

    request_url = algorithms_url + "/demo"
    request_body = {
        "inputdata": {
            "pathology": "dementia",
            "dataset": ["adni"],
            "filter": {},
            "x": ["lefthippocampus", "righthippocampus"],
            "y": ["alzheimerbroadcategory"]
        },
        "parameters": {
            "my_enum_param": "a",
            "my_int_param": 3,
            "list_param": [0.8, 0.95]
        },
        "crossvalidation": {
            "type": "k_fold",
            "nsplits": 10,
            "metrics": ["precision", "confusion_matrix"]
        }
    }
    headers = {"Content-type": "application/json", "Accept": "text/plain"}

    request = requests.post(request_url, data=json.dumps(request_body), headers=headers)

    assert request.status_code == 200

    assert request.text == "Success!"


def get_user_error_response(message: str):
    return {"text/plain+user_error": message}
