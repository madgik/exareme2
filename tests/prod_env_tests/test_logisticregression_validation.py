from pathlib import Path

import numpy as np
import pytest

from tests.algorithm_validation_tests.helpers import algorithm_request
from tests.algorithm_validation_tests.helpers import assert_allclose
from tests.algorithm_validation_tests.helpers import get_test_params
from tests.algorithm_validation_tests.helpers import parse_response

pytest.skip(
    allow_module_level=True,
    msg="DummyEncoder is temporarily disabled due to changes in "
    "the UDF generator API. Will be re-implemented in ticket "
    "https://team-1617704806227.atlassian.net/browse/MIP-757",
)

algorithm_name = "logistic_regression"

expected_file = Path(__file__).parent / "expected" / f"{algorithm_name}_expected.json"


@pytest.mark.parametrize(
    "test_input, expected",
    get_test_params(
        expected_file,
        skip_indices=[13],
        skip_reason="Awaiting https://team-1617704806227.atlassian.net/browse/MIP-698",
    ),
)
def test_logisticregression_algorithm(test_input, expected, subtests):
    response = algorithm_request(algorithm_name, test_input)
    result = parse_response(response)

    result = result["summary"]

    with subtests.test():
        assert result["n_obs"] == expected["n_obs"]
    with subtests.test():
        assert result["df_model"] == expected["df_model"]
    with subtests.test():
        assert result["df_resid"] == expected["df_resid"]

    with subtests.test():
        assert_allclose(result["aic"], expected["aic"])
    with subtests.test():
        assert_allclose(result["bic"], expected["bic"])
    with subtests.test():
        assert_allclose(result["ll"], expected["ll"])
    with subtests.test():
        assert_allclose(result["ll0"], expected["ll0"])
    with subtests.test():
        assert_allclose(result["r_squared_mcf"], expected["r_squared_mcf"])
    with subtests.test():
        assert_allclose(result["coefficients"], expected["coefficients"])

    # some quantities need a higher tolerance due to error propagation effects
    with subtests.test():
        np.testing.assert_allclose(
            result["stderr"],
            expected["stderr"],
            rtol=1e-5,
            atol=1e-5,
        )
    with subtests.test():
        np.testing.assert_allclose(
            result["z_scores"],
            expected["z_scores"],
            rtol=1e-5,
            atol=1e-5,
        )
    with subtests.test():
        np.testing.assert_allclose(
            result["pvalues"],
            expected["pvalues"],
            rtol=1e-5,
            atol=1e-5,
        )
    with subtests.test():
        np.testing.assert_allclose(
            result["lower_ci"],
            expected["lower_ci"],
            rtol=1e-5,
            atol=1e-5,
        )
    with subtests.test():
        np.testing.assert_allclose(
            result["upper_ci"],
            expected["upper_ci"],
            rtol=1e-5,
            atol=1e-5,
        )
