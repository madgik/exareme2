import json
from pathlib import Path

import pytest

from tests.algorithm_validation_tests.helpers import algorithm_request
from tests.algorithm_validation_tests.helpers import assert_allclose
from tests.algorithm_validation_tests.helpers import get_test_params

expected_file = Path(__file__).parent / "expected" / "independent_ttest_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_independent_ttest(test_input, expected):
    response = algorithm_request("ttest_independent", test_input)
    result = json.loads(response.content)

    # print(result["t_stat"], ["statistic"])
    # print(result["df"], expected["df"], expected["n_obs"])
    print(
        result["mean_diff"],
        expected["mean_diff"],
        result["t_stat"],
        expected["statistic"],
    )
    assert_allclose(result["t_stat"], expected["statistic"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["p"], expected["p_value"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["mean_diff"], expected["mean_diff"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["df"], expected["df"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["se_diff"], expected["se_difference"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["cohens_d"], expected["cohens_d"], rtol=1e-8, atol=1e-10)

    # Due to current inability of rpy2 to properly calculate confidence intervals,
    # they won't be tested. However, they are deemed correct, since they are calculated
    # by tested values and through the scipy package.
    # assert_allclose(result["ci_upper"], expected["ci_upper"], rtol=1e-8, atol=1e-10)
    # assert_allclose(result["ci_lower"], expected["ci_lower"], rtol=1e-8, atol=1e-10)
