import json
from pathlib import Path

import numpy as np
import pytest
import requests

expected_file = Path(__file__).parent / "expected" / "pca_expected.json"


def pca_request(input):
    url = "http://127.0.0.1:4999/algorithms" + "/pca"

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
def test_pca_algorithm(test_input, expected):
    response = pca_request(test_input)
    result = json.loads(response.content)

    assert response.status_code == 200
    assert int(result["n_obs"]) == int(expected["n_obs"])
    np.testing.assert_allclose(
        result["eigenvalues"],
        expected["eigen_vals"],
        rtol=1e-7,
        atol=1e-10,
    )
    for u, v in zip(result["eigenvectors"], expected["eigen_vecs"]):
        assert_vectors_are_collinear(u, v)


def assert_vectors_are_collinear(u, v):
    cosine_similarity = np.dot(v, u) / (np.sqrt(np.dot(v, v)) * np.sqrt(np.dot(u, u)))
    np.testing.assert_allclose(abs(cosine_similarity), 1, rtol=1e-7, atol=1e-10)
