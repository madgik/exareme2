from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exareme2.helpers import algorithm_request
from tests.algorithm_validation_tests.exareme2.helpers import assert_allclose
from tests.algorithm_validation_tests.exareme2.helpers import get_test_params
from tests.algorithm_validation_tests.exareme2.helpers import parse_response
from tests.algorithm_validation_tests.exareme2.test_descriptive_stats import (
    compare_results,
)

algorithm_name = "descriptive_stats_exaflow_aggregator"

expected_file = Path(__file__).parent / "expected" / "descriptive_stats_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_descriptive_stats_exaflow(test_input, expected):
    response = algorithm_request(algorithm_name, test_input)
    result = parse_response(response)
    compare_results(result, expected)
