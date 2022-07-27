import json
from pathlib import Path

import numpy as np
import pytest

from tests.algorithm_validation_tests.helpers import algorithm_request
from tests.algorithm_validation_tests.helpers import assert_allclose
from tests.algorithm_validation_tests.helpers import get_test_params

expected_file = Path(__file__).parent / "expected" / "one_sample_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_one_sample_ttest(test_input, expected):
    response = algorithm_request("one_sample_ttest", test_input)
    result = json.loads(response.content)

    assert_allclose(result["t_stat"], expected["t_value"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["p"], expected["p_value"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["df"], expected["df"], rtol=1e-8, atol=1e-10)
    # assert_allclose(result["mean_diff"], expected["mean_diff"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["se_diff"], expected["se_diff"], rtol=1e-8, atol=1e-10)
    # assert_allclose(result["ci_upper"], expected["ci_upper"], rtol=1e-8, atol=1e-10)
    # assert_allclose(result["ci_lower"], expected["ci_lower"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["cohens_d"], expected["cohens_d"], rtol=1e-8, atol=1e-10)
