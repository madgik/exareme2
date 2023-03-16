from pathlib import Path

import pytest

from tests.algorithm_validation_tests.helpers import algorithm_request
from tests.algorithm_validation_tests.helpers import get_test_params
from tests.algorithm_validation_tests.helpers import parse_response
from tests.algorithm_validation_tests.test_descriptive_stats import compare_results

algorithm_name = "descriptive_stats"

expected_file = Path(__file__).parent / "expected" / f"{algorithm_name}_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_descriptive_stats(test_input, expected):
    response = algorithm_request(algorithm_name, test_input)
    result = parse_response(response)

    compare_results(result, expected)
