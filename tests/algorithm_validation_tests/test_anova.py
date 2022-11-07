import json
from pathlib import Path

import numpy as np
import pytest

from tests.algorithm_validation_tests.helpers import algorithm_request
from tests.algorithm_validation_tests.helpers import assert_allclose
from tests.algorithm_validation_tests.helpers import get_test_params

expected_file = Path(__file__).parent / "expected" / "anova_twoway_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_anova_two_way(test_input, expected):
    response = algorithm_request("anova", test_input)
    result = json.loads(response.content)

    assert_allclose(result["n_obs"], expected["n_obs"], rtol=1e-8, atol=1e-10)
    assert_allclose(result["grand_mean"], expected["mean_y"], rtol=1e-8, atol=1e-10)

    x1_label = test_input["inputdata"]["x"][0]
    x2_label = test_input["inputdata"]["x"][1]
    for key, e_val in expected.items():
        if key == "sum_sq":
            assert e_val[f"{x1_label}"] == result["sum_sq_x1"]
            assert e_val[f"{x2_label}"] == result["sum_sq_x2"]
