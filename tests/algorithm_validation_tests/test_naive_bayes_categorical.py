from pathlib import Path

import pytest

from tests.algorithm_validation_tests.helpers import get_test_params
from tests.algorithm_validation_tests.test_naive_bayes_gaussian import get_test_inputs

fit_exp = Path(__file__).parent / "expected" / "naive_bayes_categorical_fit.json"
pred_exp = Path(__file__).parent / "expected" / "naive_bayes_categorical_predict.json"
cv_inputs = Path(__file__).parent / "expected" / "naive_bayes_categorical_cv.json"


class TestCategoricalNB:
    @pytest.mark.parametrize("test_input, expected", get_test_params(fit_exp))
    def test_fit__class_count(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result("test_nb_categorical_fit", test_input)
        assert result["class_count"] == expected["class_count"]

    @pytest.mark.parametrize("test_input, expected", get_test_params(fit_exp))
    def test_fit__category_count(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result("test_nb_categorical_fit", test_input)
        res_cc = result["category_count"]
        exp_cc = expected["category_count"]

        self._assert_category_count_match(res_cc, exp_cc)

    @pytest.mark.parametrize("test_input, expected", get_test_params(pred_exp))
    def test_predict(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result("test_nb_categorical_predict", test_input)
        assert result["predictions"] == expected["predictions"]

    @pytest.mark.parametrize("test_input", get_test_inputs(cv_inputs))
    def test_cv(self, test_input, get_algorithm_result):
        result = get_algorithm_result("naive_bayes_categorical_cv", test_input)
        assert result

    @staticmethod
    def _assert_category_count_match(res, exp):
        for res_feat, exp_feat in zip(res, exp):
            for res_cls, exp_cls in zip(res_feat, exp_feat):
                assert set(res_cls) == set(exp_cls)
