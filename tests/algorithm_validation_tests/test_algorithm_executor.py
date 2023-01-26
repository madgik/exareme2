import json

import pytest
import requests

from mipengine.controller.api.error_handlers import HTTPStatusCode

algorithm_name = "logistic_regression_cv"


@pytest.fixture
def input_no_node_with_sufficient_data():
    # with this request local nodes: localnode4,localnode1,localnode3,localnode2,localnode5
    # will be initially chosen but then after data model views are created none of the
    # nodes will have sufficient data so the algorithm will not continue executimng
    filters = {
        "condition": "AND",
        "rules": [
            {
                "id": "dataset",
                "type": "string",
                "value": [
                    "edsd0",
                    "edsd1",
                    "edsd2",
                    "edsd3",
                    "edsd4",
                    "edsd5",
                    "edsd6",
                    "edsd7",
                    "edsd8",
                    "edsd9",
                ],
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
    }

    request = {
        "inputdata": {
            "y": ["parkinsonbroadcategory"],
            "x": [
                "cerebellarvermallobulesiv",
                "rightainsanteriorinsula",
                "leftventraldc",
                "rightcerebellumwhitematter",
                "leftfrpfrontalpole",
                "rightpcggposteriorcingulategyrus",
                "rightofugoccipitalfusiformgyrus",
                "leftputamen",
                "leftscasubcallosalarea",
                "rightangangulargyrus",
                "rightgregyrusrectus",
                "leftpcuprecuneus",
                "subjectageyears",
                "leftpogpostcentralgyrus",
                "leftphgparahippocampalgyrus",
                "rightmsfgsuperiorfrontalgyrusmedialsegment",
            ],
            "data_model": "dementia:0.1",
            "datasets": [
                "desd-synthdata8",
                "desd-synthdata7",
                "edsd8",
                "ppmi2",
                "edsd1",
                "ppmi3",
                "desd-synthdata0",
                "ppmi1",
                "ppmi7",
            ],
            "filters": filters,
            "valid": True,
        },
        "parameters": {"positive_class": "PD", "n_splits": 6},
        "test_case_num": 9,
    }
    return request


@pytest.fixture
def input_subset_of_nodes_has_sufficient_data():
    # with this request local nodes: localnode4,localnode1,localnode5,localnode2,localnode3
    # will be initially chosen but then after data model views are created, nodes:
    # localnode1 and localnode3 will be removed and the algorithm will continue executing
    # on nodes: localnode4, localnode5, localnode2
    request = {
        "inputdata": {
            "y": ["parkinsonbroadcategory"],
            "x": [
                "cerebellarvermallobulesiv",
                "rightainsanteriorinsula",
                "leftventraldc",
                "rightcerebellumwhitematter",
                "leftfrpfrontalpole",
                "rightpcggposteriorcingulategyrus",
                "rightofugoccipitalfusiformgyrus",
                "leftputamen",
                "leftscasubcallosalarea",
                "rightangangulargyrus",
                "rightgregyrusrectus",
                "leftpcuprecuneus",
                "subjectageyears",
                "leftpogpostcentralgyrus",
                "leftphgparahippocampalgyrus",
                "rightmsfgsuperiorfrontalgyrusmedialsegment",
            ],
            "data_model": "dementia:0.1",
            "datasets": [
                "desd-synthdata8",
                "desd-synthdata7",
                "edsd8",
                "ppmi2",
                "edsd1",
                "ppmi3",
                "desd-synthdata0",
                "ppmi1",
                "ppmi7",
            ],
            "filters": None,
            "valid": True,
        },
        "parameters": {"positive_class": "PD", "n_splits": 6},
        "test_case_num": 9,
    }
    return request


def test_exec_algorithm_removing_nodes_after_create_data_model_views(
    input_subset_of_nodes_has_sufficient_data,
):
    response = algorithm_request(
        algorithm_name, input_subset_of_nodes_has_sufficient_data
    )
    assert response.status_code == HTTPStatusCode.OK


def test_algo_execution_stops_after_create_data_model_view_all_nodes_removed(
    input_no_node_with_sufficient_data,
):
    response = algorithm_request(algorithm_name, input_no_node_with_sufficient_data)
    assert response.status_code == HTTPStatusCode.INSUFFICIENT_DATA_ERROR


def algorithm_request(algorithm: str, input: dict):
    url = "http://127.0.0.1:5000/algorithms" + f"/{algorithm}"
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(url, data=json.dumps(input), headers=headers)
    return response
