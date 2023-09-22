from pathlib import Path

import pytest

from tests.algorithm_validation_tests.helpers import algorithm_request
from tests.algorithm_validation_tests.helpers import get_test_params
from tests.algorithm_validation_tests.helpers import parse_response
from tests.algorithm_validation_tests.test_anova import validate_results

expected_file = Path(__file__).parent / "expected" / "anova_twoway_prod_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_anova_two_way(test_input, expected, subtests):
    response = algorithm_request("anova", test_input)
    result = parse_response(response)

    validate_results(result, expected, subtests)
