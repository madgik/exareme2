from pathlib import Path

import numpy as np
import pytest

from tests.algorithm_validation_tests.exareme2.helpers import get_test_params
from tests.algorithm_validation_tests.exareme2.test_naive_bayes_gaussian import (
    get_test_inputs,
)

fit_exp = (
    Path(__file__).parent / "expected" / "naive_bayes_categorical_fit_expected.json"
)
pred_exp = (
    Path(__file__).parent / "expected" / "naive_bayes_categorical_predict_expected.json"
)
cv_inputs = (
    Path(__file__).parent / "expected" / "naive_bayes_categorical_cv_expected.json"
)


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
    def test_cv__confusion_matrix(self, test_input, get_algorithm_result):
        result = get_algorithm_result("naive_bayes_categorical_cv", test_input)
        confmat = np.array(result["confusion_matrix"]["data"])
        assert (0 <= confmat).all()

    @pytest.mark.parametrize("test_input", get_test_inputs(cv_inputs))
    def test_cv__accuracy(self, test_input, get_algorithm_result):
        result = get_algorithm_result("naive_bayes_categorical_cv", test_input)
        accuracy = result["classification_summary"]["accuracy"]
        self._assert_all_between_0_and_1(accuracy)

    @pytest.mark.parametrize("test_input", get_test_inputs(cv_inputs))
    def test_cv__precision(self, test_input, get_algorithm_result):
        result = get_algorithm_result("naive_bayes_categorical_cv", test_input)
        precision = result["classification_summary"]["precision"]
        self._assert_all_between_0_and_1(precision)

    @pytest.mark.parametrize("test_input", get_test_inputs(cv_inputs))
    def test_cv__recall(self, test_input, get_algorithm_result):
        result = get_algorithm_result("naive_bayes_categorical_cv", test_input)
        recall = result["classification_summary"]["recall"]
        self._assert_all_between_0_and_1(recall)

    @pytest.mark.parametrize("test_input", get_test_inputs(cv_inputs))
    def test_cv__fscore(self, test_input, get_algorithm_result):
        result = get_algorithm_result("naive_bayes_categorical_cv", test_input)
        fscore = result["classification_summary"]["fscore"]
        self._assert_all_between_0_and_1(fscore)

    @staticmethod
    def _assert_category_count_match(res, exp):
        for res_feat, exp_feat in zip(res, exp):
            for res_cls, exp_cls in zip(res_feat, exp_feat):
                assert set(res_cls) == set(exp_cls)

    @staticmethod
    def _assert_all_between_0_and_1(quantity):
        for key1, val1 in quantity.items():
            for key2, val2 in val1.items():
                msg = f"{key1}:{key2} is out of bounds"
                assert 0 <= val2 <= 1, msg
