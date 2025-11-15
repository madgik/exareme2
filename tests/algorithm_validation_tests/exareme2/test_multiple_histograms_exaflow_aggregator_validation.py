from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exareme2.helpers import algorithm_request
from tests.algorithm_validation_tests.exareme2.helpers import get_test_params
from tests.algorithm_validation_tests.exareme2.helpers import parse_response

algorithm_name = "multiple_histograms_exaflow_aggregator"

expected_file = Path(__file__).parent / "expected" / "multiple_histograms_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_histogram_exaflow(test_input, expected):
    response = algorithm_request(algorithm_name, test_input)
    result = parse_response(response)
    assert expected["histogram"] == result["histogram"]
