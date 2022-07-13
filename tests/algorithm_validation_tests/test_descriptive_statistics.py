import json
from pathlib import Path

import numpy as np
import pytest
import requests

expected_file = Path(__file__).parent / "expected" / "descriptive_statistics_expected.json"


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
    assert result["categorical_counts"] == expected["categorical_counts"]
    assert result["categorical_variables"] == expected["categorical_columns"]
    assert result["numerical_variables"] == expected["numerical_columns"]
    np.testing.assert_allclose(
        result["max_model"],
        expected["max_model"],
        rtol=1e-7,
        atol=1e-10,
    )
