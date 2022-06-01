import functools
import json
from pathlib import Path

import numpy as np
import pytest

from tests.algorithm_validation_tests.helpers import algorithm_request
from tests.algorithm_validation_tests.helpers import get_test_params

expected_file = Path(__file__).parent / "expected" / "linear_regression_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_linearregression_algorithm(test_input, expected):
    response = algorithm_request("linear_regression", test_input)
    try:
        result = json.loads(response.text)
    except json.decoder.JSONDecodeError:
        print(response.text)
        raise
    assert_allclose = functools.partial(
        np.testing.assert_allclose,
        rtol=1e-7,
        atol=1e-10,
    )

    assert result["n_obs"] == expected["n_obs"]
    assert result["df_resid"] == expected["df_resid"]
    assert result["df_model"] == expected["df_model"]
    assert_allclose(result["coefficients"], expected["coefficients"])
    assert_allclose(result["std_err"], expected["std_err"])
    assert_allclose(result["t_stats"], expected["t_stat"])
    assert_allclose(result["pvalues"], expected["t_p_values"])
    assert_allclose(result["lower_ci"], expected["lower_ci"])
    assert_allclose(result["upper_ci"], expected["upper_ci"])
    assert_allclose(result["rse"], expected["rse"])
    assert_allclose(result["r_squared"], expected["r_squared"])
    assert_allclose(result["r_squared_adjusted"], expected["r_squared_adjusted"])
    assert_allclose(result["f_stat"], expected["f_stat"])
    assert_allclose(result["f_pvalue"], expected["f_p_value"])
