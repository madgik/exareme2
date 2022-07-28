import copy
import json
from pathlib import Path

import numpy as np
import pytest
import requests

expected_file = Path(__file__).parent / "expected" / "pearson_expected.json"


def pearson_request(input):
    url = "http://127.0.0.1:5000/algorithms" + "/pearson"

    variables = copy.deepcopy(input["inputdata"]["y"])
    if input["inputdata"]["x"]:
        variables.extend(input["inputdata"]["x"])

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
def test_pearson_algorithm(test_input, expected):
    response = pearson_request(test_input)
    try:
        result = json.loads(response.content)
    except json.decoder.JSONDecodeError as exc:
        print(response)
        raise exc
    assert response.status_code == 200
    assert int(result["n_obs"]) == int(expected["n_obs"])
    for var in test_input["inputdata"]["y"]:
        np.testing.assert_allclose(
            result["correlations"][f"{var}"],
            expected["correlations"][f"{var}"],
            rtol=1e-7,
            atol=1e-10,
        )

        np.testing.assert_allclose(
            result["p_values"][f"{var}"],
            expected["p-values"][f"{var}"],
            rtol=1e-7,
            atol=1e-10,
        )

        np.testing.assert_allclose(
            result["ci_lo"][f"{var}"],
            expected["low_confidence_intervals"][f"{var}"],
            rtol=1e-7,
            atol=1e-10,
        )

        np.testing.assert_allclose(
            result["ci_hi"][f"{var}"],
            expected["high_confidence_intervals"][f"{var}"],
            rtol=1e-7,
            atol=1e-10,
        )


def assert_vectors_are_collinear(u, v):
    cosine_similarity = np.dot(v, u) / (np.sqrt(np.dot(v, v)) * np.sqrt(np.dot(u, u)))
    np.testing.assert_allclose(abs(cosine_similarity), 1, rtol=1e-7, atol=1e-10)
