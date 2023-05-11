import json
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from tests.algorithm_validation_tests.helpers import algorithm_request
from tests.algorithm_validation_tests.helpers import get_test_params
from tests.algorithm_validation_tests.helpers import parse_response

assert_allclose = partial(np.testing.assert_allclose, rtol=1e-2, atol=1e-2)
gnb_fit_exp = Path(__file__).parent / "expected" / "naive_bayes_gauss_fit.json"
gnb_pred_exp = Path(__file__).parent / "expected" / "naive_bayes_gauss_predict.json"
gnb_cv_inputs = Path(__file__).parent / "expected" / "naive_bayes_gauss_cv.json"

gnb_fit_test_params = get_test_params(gnb_fit_exp)
gnb_pred_test_params = get_test_params(gnb_pred_exp)


@pytest.fixture(scope="class")
def get_result():
    cache = {}

    def _get_result(algname, test_input):
        test_case_num = test_input["test_case_num"]
        key = (algname, test_case_num)
        if key not in cache:
            response = algorithm_request(algname, test_input)
            result = parse_response(response)
            cache[key] = result
        return cache[key]

    return _get_result


class TestGaussianNBFit:
    @pytest.mark.parametrize("test_input, expected", gnb_fit_test_params)
    def test_class_count(self, test_input, expected, get_result):
        result = get_result("test_nb_gaussian_fit", test_input)
        assert result["class_count"] == expected["class_count"]

    @pytest.mark.parametrize("test_input, expected", gnb_fit_test_params)
    def test_theta(self, test_input, expected, get_result):
        result = get_result("test_nb_gaussian_fit", test_input)
        assert_allclose(result["theta"], expected["theta"])

    @pytest.mark.parametrize("test_input, expected", gnb_fit_test_params)
    def test_var(self, test_input, expected, get_result):
        result = get_result("test_nb_gaussian_fit", test_input)
        assert_allclose(result["var"], expected["var"])


class TestGaussianNBPredict:
    @pytest.mark.parametrize("test_input, expected", gnb_pred_test_params)
    def test_predictions(self, test_input, expected):
        response = algorithm_request("test_nb_gaussian_predict", test_input)
        result = parse_response(response)

        assert result["predictions"] == expected["predictions"]


def get_test_inputs(file):
    with file.open() as f:
        params = json.load(f)["test_cases"]
    return [p["input"] for p in params]


@pytest.mark.parametrize("test_input", get_test_inputs(gnb_cv_inputs))
def test_naive_bayes_gaussian_cv(test_input, get_result):
    response = algorithm_request("naive_bayes_gaussian_cv", test_input)
    result = parse_response(response)

    assert result
