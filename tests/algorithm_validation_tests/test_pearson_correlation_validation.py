import copy
import json
from pathlib import Path

import numpy as np
import pytest
import requests

from tests.algorithm_validation_tests.helpers import algorithm_request

expected_file = Path(__file__).parent / "expected" / "pearson_correlation_expected.json"


def get_test_params(expected_file, slc=None):
    with expected_file.open() as f:
        params = json.load(f)["test_cases"]
    if not slc:
        slc = slice(len(params))
    params = [(p["input"], p["output"]) for p in params[slc]]
    return params


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_pearson_algorithm(test_input, expected):
    response = algorithm_request("pearson_correlation", test_input)
    result = json.loads(response.content)

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
