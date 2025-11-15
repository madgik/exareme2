from pathlib import Path

import pytest

from tests.algorithm_validation_tests.exareme2.helpers import algorithm_request
from tests.algorithm_validation_tests.exareme2.helpers import assert_allclose
from tests.algorithm_validation_tests.exareme2.helpers import get_test_params
from tests.algorithm_validation_tests.exareme2.helpers import parse_response

algorithm_name = "ttest_paired_exaflow_aggregator"

expected_file = Path(__file__).parent / "expected" / "ttest_paired_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_paired_ttest_exaflow(test_input, expected):
    response = algorithm_request(algorithm_name, test_input)
    result = parse_response(response)

    assert_allclose(result["t_stat"], expected["statistic"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["p"], expected["p_value"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["df"], expected["df"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["mean_diff"], expected["mean_diff"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["se_diff"], expected["se_difference"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["cohens_d"], expected["cohens_d"], rtol=1e-8, atol=1e-10)
