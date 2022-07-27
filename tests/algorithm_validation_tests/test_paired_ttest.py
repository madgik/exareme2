import json
from pathlib import Path

import numpy as np
import pytest

from tests.algorithm_validation_tests.helpers import algorithm_request
from tests.algorithm_validation_tests.helpers import assert_allclose
from tests.algorithm_validation_tests.helpers import get_test_params

expected_file = Path(__file__).parent / "expected" / "paired_ttest_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_paired_ttest(test_input, expected):
    response = algorithm_request("paired_ttest", test_input)
    result = json.loads(response.content)

    print(
        f"res_upper: {result['ci_upper']}, exp_upper: {expected['ci_upper']}, res_lower: {result['ci_lower']}, exp_lower: {expected['ci_lower']}"
    )
    assert_allclose(result["t_stat"], expected["statistic"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["p"], expected["p_value"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["df"], expected["df"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["mean_diff"], expected["mean_diff"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["se_diff"], expected["se_difference"], rtol=1e-8, atol=1e-10)
    # assert_allclose(result["ci_upper"], expected["ci_upper"], rtol=1e-8, atol=1e-10)
    # assert_allclose(result["ci_lower"], expected["ci_lower"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["cohens_d"], expected["cohens_d"], rtol=1e-8, atol=1e-10)
