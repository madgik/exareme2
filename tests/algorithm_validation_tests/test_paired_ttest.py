import json
from pathlib import Path

import numpy as np
import pytest
import requests

expected_file = Path(__file__).parent / "expected" / "paired_ttest_expected.json"


def paired_ttest_request(input):
    url = "http://127.0.0.1:5000/algorithms" + "/paired_ttest"

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
def test_paired_ttest(test_input, expected):
    response = paired_ttest_request(test_input)
    result = json.loads(response.content)

    assert np.isclose(result["t_stat"], expected["statistic"])
    assert np.isclose(result["p"], expected["p_value"])
    assert np.isclose(result["df"], expected["df"])
    assert np.isclose(result["mean_diff"], expected["mean_diff"])
    assert np.isclose(result["se_diff"], expected["se_difference"])
    assert np.isclose(result["ci_upper"], expected["ci_upper"], rtol=1e-8, atol=1e-10)
    assert np.isclose(result["ci_lower"], expected["ci_lower"], rtol=1e-8, atol=1e-10)
    assert np.isclose(result["cohens_d"], expected["cohens_d"], rtol=1e-1, atol=1e-10)
