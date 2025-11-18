import pytest

from tests.algorithm_validation_tests.exaflow import algorithm_request
from tests.algorithm_validation_tests.exaflow import parse_response
from tests.algorithm_validation_tests.exaflow.test_naive_bayes_categorical import (
    cv_inputs,
)
from tests.algorithm_validation_tests.exaflow.test_naive_bayes_gaussian import (
    get_test_inputs,
)


@pytest.mark.parametrize("test_input", get_test_inputs(cv_inputs))
def test_naive_bayes_gaussian_cv(test_input):
    response = algorithm_request("naive_bayes_categorical_cv", test_input)
    result = parse_response(response)

    assert result
