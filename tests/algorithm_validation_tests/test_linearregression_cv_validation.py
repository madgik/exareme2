"""
Cross validation is non deterministic because it works by randomly splitting
the datasets into train/test sets. This splitting is taken care of by sklearn
in the state-of-the-art implementation and it's impossible to control or
replicate in our federated implementation. For this reason the tests cannot be
deterministic and we have to resort to statistical testing. In what follows,
some quantity is computed for each CV fold. The way we test for equality
between our and the SOA implementation is by conducting a two-tailed t-test
with 99% confidence.
"""
import json
from pathlib import Path

import pytest
import scipy.stats as st

from tests.algorithm_validation_tests.helpers import algorithm_request
from tests.algorithm_validation_tests.helpers import get_test_params

algorithm_name = "linear_regression_cv"

expected_file = Path(__file__).parent / "expected" / f"{algorithm_name}_expected.json"


@pytest.fixture(scope="module")
def cache():
    yield {}


def get_cached_response(algorithm_name, test_input, cache):
    test_case_num = test_input["test_case_num"]
    if test_case_num in cache:
        response = cache[test_case_num]
    else:
        response = algorithm_request(algorithm_name, test_input)
        cache[test_case_num] = response
    return response


@pytest.mark.parametrize(
    "test_input, expected",
    get_test_params(
        expected_file,
        skip_indices=[4, 27, 39, 40],
        skip_reason="Run on five nodes in CI results in empty tables,"
        " see https://team-1617704806227.atlassian.net/browse/MIP-634",
    ),
)
def test_linearregression_cv_non_inferiority_msre(test_input, expected, cache):
    response = get_cached_response(algorithm_name, test_input, cache)
    result = json.loads(response.content)

    n_splits = test_input["parameters"]["n_splits"]
    msre_res_mean, msre_res_std = result["mean_sq_error"]
    msre_exp_mean, msre_exp_std = expected["mean_sq_error"]
    ttest = st.ttest_ind_from_stats(
        msre_res_mean,
        msre_res_std,
        n_splits,
        msre_exp_mean,
        msre_exp_std,
        n_splits,
        equal_var=True,
        alternative="two-sided",
    )
    assert ttest.pvalue >= 0.01


@pytest.mark.parametrize(
    "test_input, expected",
    get_test_params(
        expected_file,
        skip_indices=[4, 27, 39, 40],
        skip_reason="Run on five nodes in CI results in empty tables,"
        " see https://team-1617704806227.atlassian.net/browse/MIP-634",
    ),
)
def test_linearregression_cv_non_inferiority_mae(test_input, expected, cache):
    response = get_cached_response("linear_regression_cv", test_input, cache)
    try:
        result = json.loads(response.text)
    except json.decoder.JSONDecodeError:
        raise ValueError(f"The result is not valid json:\n{response.text}") from None

    n_splits = test_input["parameters"]["n_splits"]
    mae_res_mean, mae_res_std = result["mean_abs_error"]
    mae_exp_mean, mae_exp_std = expected["mean_abs_error"]
    ttest = st.ttest_ind_from_stats(
        mae_res_mean,
        mae_res_std,
        n_splits,
        mae_exp_mean,
        mae_exp_std,
        n_splits,
        equal_var=True,
        alternative="two-sided",
    )
    assert ttest.pvalue >= 0.01


@pytest.mark.parametrize(
    "test_input, expected",
    get_test_params(
        expected_file,
        skip_indices=[4, 27, 39, 40],
        skip_reason="Run on five nodes in CI results in empty tables,"
        " see https://team-1617704806227.atlassian.net/browse/MIP-634",
    ),
)
def test_linearregression_cv_non_inferiority_r2(test_input, expected, cache):
    response = get_cached_response("linear_regression_cv", test_input, cache)
    try:
        result = json.loads(response.text)
    except json.decoder.JSONDecodeError:
        raise ValueError(f"The result is not valid json:\n{response.text}") from None

    n_splits = test_input["parameters"]["n_splits"]
    r2_res_mean, r2_res_std = result["r_squared"]
    r2_exp_mean, r2_exp_std = expected["r_squared"]
    ttest = st.ttest_ind_from_stats(
        r2_res_mean,
        r2_res_std,
        n_splits,
        r2_exp_mean,
        r2_exp_std,
        n_splits,
        equal_var=True,
        alternative="two-sided",
    )
    assert ttest.pvalue >= 0.01
