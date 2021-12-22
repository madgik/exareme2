import pytest
import json
import numpy as np
from mipengine.algorithms.requests.pca_request import do_post_request
from pathlib import Path

expected_file = Path(__file__).parent / "expected" / "pca_expected.json"


def get_test_params(expected_file, slc=None):
    with expected_file.open() as f:
        params = json.load(f)["test_cases"]
    if not slc:
        slc = slice(len(params))
        print(f"{slc =}")
    params = [(p["input"], p["output"]) for p in params[slc]]
    return params


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_pca_algorithm_local(test_input, expected):
    response = do_post_request(test_input)
    result = json.loads(response.content)

    assert response.status_code == 200
    assert int(result["n_obs"]) == int(expected["n_obs"])
    assert np.isclose(result["eigenvalues"], expected["eigen_vals"], atol=1e-2).all()
    for u, v in zip(result["eigenvectors"], expected["eigen_vecs"]):
        assert are_collinear(u, v)


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_pca_algorithm_federated(test_input, expected):
    response = do_post_request(test_input)
    result = json.loads(response.content)

    # assert int(result["n_obs"]) == int(expected["n_obs"])
    assert np.isclose(result["eigenvalues"], expected["eigen_vals"], atol=1e-2).all()
    for u, v in zip(result["eigenvectors"], expected["eigen_vecs"]):
        assert are_collinear(u, v)


def are_collinear(u, v):
    cosine_similarity = np.dot(v, u) / (np.sqrt(np.dot(v, v)) * np.sqrt(np.dot(u, u)))
    return np.isclose(abs(cosine_similarity), 1, rtol=1e-5)
