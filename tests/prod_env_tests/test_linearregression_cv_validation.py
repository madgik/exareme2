from pathlib import Path

import pytest

from tests.algorithm_validation_tests.helpers import algorithm_request
from tests.algorithm_validation_tests.helpers import get_test_params
from tests.algorithm_validation_tests.helpers import parse_response

expected_file = (
    Path(__file__).parent / "expected" / "linear_regression_cv_expected.json"
)


@pytest.mark.parametrize("test_input, _", get_test_params(expected_file))
def test_linear_regression_cv(test_input, _):
    response = algorithm_request("linear_regression_cv", test_input)
    result = parse_response(response)
    assert result
