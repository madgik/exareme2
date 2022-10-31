import json
from pathlib import Path

import numpy as np
import pytest
import requests

from tests.algorithm_validation_tests.helpers import algorithm_request

expected_file = Path(__file__).parent / "expected" / "pca_expected.json"


def get_test_params(expected_file, slc=None):
    with expected_file.open() as f:
        params = json.load(f)["test_cases"]
    if not slc:
        slc = slice(len(params))
    params = [(p["input"], p["output"]) for p in params[slc]]
    return params


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_pca_algorithm(test_input, expected):
    response = algorithm_request("pca", test_input)

    if response.status_code != 200:
        raise ValueError(
            f"Unexpected response status: '{response.status_code}'. Response message: '{response.content}'"
        )
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
