import json
from pathlib import Path

import numpy as np
import pytest
import requests

expected_file = Path(__file__).parent / "expected" / "descriptive2.json"


def desc_request(input):
    url = "http://127.0.0.1:5000/algorithms" + "/descriptive_statistics"

    filters = None
    input["inputdata"]["filters"] = filters
    request_json = json.dumps(input)

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(url, data=request_json, headers=headers)
    return response


def get_test_params(expected_file, slc=None):
    with expected_file.open() as f:
        params = json.load(f)["test_cases"]
    if not slc:
        slc = slice(len(params))
    params = [(p["input"], p["output"]) for p in params[slc]]
    return params


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_descriptive_algorithm(test_input, expected):
    response = desc_request(test_input)
    result = json.loads(response.content)

    assert response.status_code == 200
    """
    assert result["categorical_counts"] == expected["categorical_counts"]
    assert result["categorical_variables"] == expected["categorical_columns"]
    assert result["numerical_variables"] == expected["numerical_columns"]
    np.testing.assert_allclose(
        result["max_model"],
        expected["max_model"],
        rtol=1e-7,
        atol=1e-10,
    )
    """
    numerical_variables = expected['numerical_columns']
    categorical_variables = expected['categorical_columns']

    results_single = result['single']
    results_model = result['model']

    max_single = data_to_list_numerical(numerical_variables,'global','max',results_single)
    max_model = data_to_list_numerical(numerical_variables,'global','max',results_model)

    categoricals_counts = data_to_list_categorical(categorical_variables,'global',results_single)


    np.testing.assert_allclose(
        max_model,
        expected["max_model"],
        rtol=1e-7,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        max_single,
        expected["max"],
        rtol=1e-7,
        atol=1e-10,
    )

    assert  categoricals_counts == expected["categorical_counts"]

def get_data(name,dataset_name,response_list):
    for curr_response in response_list:
        if curr_response['name'] == name and curr_response['dataset_name'] == dataset_name:
            return curr_response['results']['data']

def data_to_list_numerical(variable_list,dataset_name,curr_metric,response_list):
    curr_list = []
    for curr_variable in variable_list:
        curr_data = get_data(curr_variable,dataset_name,response_list)
        curr_result = curr_data[curr_metric]
        curr_list.append(curr_result)
    return curr_list

def data_to_list_categorical(variable_list,dataset_name,response_list):
    curr_list = []
    for curr_variable in variable_list:
        curr_data = get_data(curr_variable,dataset_name,response_list)
        curr_result = curr_data
        curr_list.append(curr_result)
    return curr_list
