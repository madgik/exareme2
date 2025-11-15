from pathlib import Path

import numpy as np
import pytest

from tests.algorithm_validation_tests.exareme2.helpers import algorithm_request
from tests.algorithm_validation_tests.exareme2.helpers import get_test_params
from tests.algorithm_validation_tests.exareme2.helpers import parse_response

algorithm_name = "pearson_correlation_exaflow_aggregator"

expected_file = Path(__file__).parent / "expected" / "pearson_correlation_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_pearson_correlation_exaflow(test_input, expected):
    response = algorithm_request(algorithm_name, test_input)
    result = parse_response(response)

    assert int(result["n_obs"]) == int(expected["n_obs"])
    for var in test_input["inputdata"]["y"]:
        np.testing.assert_allclose(
            result["correlations"][f"{var}"],
            expected["correlations"][f"{var}"],
            rtol=1e-5,
            atol=1e-8,
        )

        np.testing.assert_allclose(
            result["p_values"][f"{var}"],
            expected["p-values"][f"{var}"],
            rtol=1e-5,
            atol=1e-8,
        )

        np.testing.assert_allclose(
            result["ci_lo"][f"{var}"],
            expected["low_confidence_intervals"][f"{var}"],
            rtol=1e-5,
            atol=1e-8,
        )

        np.testing.assert_allclose(
            result["ci_hi"][f"{var}"],
            expected["high_confidence_intervals"][f"{var}"],
            rtol=1e-5,
            atol=1e-8,
        )
