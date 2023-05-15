import json
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from tests.algorithm_validation_tests.helpers import get_test_params

assert_allclose = partial(np.testing.assert_allclose, rtol=1e-2, atol=1e-2)
fit_exp = Path(__file__).parent / "expected" / "naive_bayes_gauss_fit.json"
pred_exp = Path(__file__).parent / "expected" / "naive_bayes_gauss_predict.json"
cv_inputs = Path(__file__).parent / "expected" / "naive_bayes_gauss_cv.json"


def get_test_inputs(file):
    with file.open() as f:
        params = json.load(f)["test_cases"]
    return [p["input"] for p in params]


class TestGaussianNB:
    @pytest.mark.parametrize("test_input, expected", get_test_params(fit_exp))
    def test_fit__class_count(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result("test_nb_gaussian_fit", test_input)
        assert result["class_count"] == expected["class_count"]

    @pytest.mark.parametrize("test_input, expected", get_test_params(fit_exp))
    def test_fit__theta(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result("test_nb_gaussian_fit", test_input)
        assert_allclose(result["theta"], expected["theta"])

    @pytest.mark.parametrize("test_input, expected", get_test_params(fit_exp))
    def test_fit__var(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result("test_nb_gaussian_fit", test_input)
        assert_allclose(result["var"], expected["var"])

    @pytest.mark.parametrize("test_input, expected", get_test_params(pred_exp))
    def test_predict(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result("test_nb_gaussian_predict", test_input)
        assert result["predictions"] == expected["predictions"]

    @pytest.mark.parametrize("test_input", get_test_inputs(cv_inputs))
    def test_cv(self, test_input, get_algorithm_result):
        result = get_algorithm_result("naive_bayes_gaussian_cv", test_input)
        assert result
