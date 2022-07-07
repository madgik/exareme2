import json

import numpy as np
import pytest
import requests

from tests.standalone_tests.conftest import ALGORITHMS_URL


def get_parametrization_list_success_cases():
    parametrization_list = []
    # ~~~~~~~~~~success case 1~~~~~~~~~~
    algorithm_name = "smpc_standard_deviation"
    request_dict = {
        "inputdata": {
            "data_model": "dementia:0.1",
            "datasets": [
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
            "y": [
                "lefthippocampus",
            ],
            "filters": {
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
                            ]
                        ],
                    },
                ],
                "valid": True,
            },
        },
    }
    expected_response = {
        "title": "Standard Deviation",
        "columns": [
            {"name": "variable", "data": ["lefthippocampus"], "type": "STR"},
            {"name": "std_deviation", "data": [0.3634506955662605], "type": "FLOAT"},
            {"name": "min_value", "data": [1.3047], "type": "FLOAT"},
            {"name": "max_value", "data": [4.4519], "type": "FLOAT"},
        ],
    }
    parametrization_list.append((algorithm_name, request_dict, expected_response))
    # END ~~~~~~~~~~success case 1~~~~~~~~~~

    return parametrization_list


@pytest.mark.parametrize(
    "algorithm_name, request_dict, expected_response",
    get_parametrization_list_success_cases(),
)
def test_post_testing_algorithms(
    algorithm_name,
    request_dict,
    expected_response,
    localnode1_node_service,
    load_data_localnode1,
    localnode2_node_service,
    load_data_localnode2,
    localnodetmp_node_service,
    load_data_localnodetmp,
    globalnode_node_service,
    controller_service,
):

    algorithm_url = ALGORITHMS_URL + "/" + algorithm_name

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        algorithm_url,
        data=json.dumps(request_dict),
        headers=headers,
    )
    assert response.status_code == 200, f"Response message: {response.text}"

    response = response.json()
    columns = response["columns"]
    expected_columns = expected_response["columns"]

    for column, expected_column in zip(columns, expected_columns):
        assert column["name"] == expected_column["name"]
        assert column["type"] == expected_column["type"]
        if column["type"] == "STR":
            assert column["data"] == expected_column["data"]
        elif column["type"] == "FLOAT":
            np.testing.assert_allclose(column["data"], expected_column["data"])
