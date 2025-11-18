from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exaflow import algorithm_request
from tests.algorithm_validation_tests.exaflow import get_test_params
from tests.algorithm_validation_tests.exaflow import parse_response

expected_file = (
    Path(__file__).parent / "expected" / "logistic_regression_cv_expected.json"
)


@pytest.mark.parametrize("test_input, _", get_test_params(expected_file))
def test_logistic_regression_cv(test_input, _):
    response = algorithm_request("logistic_regression_cv", test_input)
    result = parse_response(response)
    assert result
