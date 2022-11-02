import json
from pathlib import Path

import pytest

from tests.algorithm_validation_tests.helpers import algorithm_request
from tests.algorithm_validation_tests.helpers import assert_allclose
from tests.algorithm_validation_tests.helpers import get_test_params

expected_file = Path(__file__).parent / "expected" / "anova_twoway_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_paired_ttest(test_input, expected):
    response = algorithm_request("anova", test_input)
    result = json.loads(response.content)

    assert_allclose(result["n_obs"], expected["n_obs"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["sum_sq"], expected["sum_sq"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["df"], expected["df"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["p_value"], expected["p_value"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["f_stat"], expected["f_stat"], rtol=1e-8, atol=1e-10)
