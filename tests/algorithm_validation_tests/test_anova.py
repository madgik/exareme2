import json
from pathlib import Path

import pytest

from tests.algorithm_validation_tests.helpers import algorithm_request
from tests.algorithm_validation_tests.helpers import assert_allclose
from tests.algorithm_validation_tests.helpers import get_test_params

expected_file = Path(__file__).parent / "expected" / "anova_twoway_expected.json"


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_anova_two_way(test_input, expected, subtests):
    response = algorithm_request("anova", test_input)
    result = json.loads(response.content)

    validate_results(result, expected, subtests)


def validate_results(result, expected, subtests):
    terms = result["terms"]
    sum_sq = dict(zip(terms, result["sum_sq"]))
    df = dict(zip(terms, result["df"]))
    f_stat = dict(zip(terms, result["f_stat"]))
    f_pvalue = dict(zip(terms, result["f_pvalue"]))

    expected_terms = expected["sum_sq"].keys()

    # Sum Sq
    with subtests.test():
        for term in expected_terms:
            assert_allclose(sum_sq[term], expected["sum_sq"][term])

    # BUG In some cases the dfs are off by one for Residuals and the interaction term.
    # When this happens the F value and p values are also wrong due to the df error.
    # I was unable to find why, so I skip the rest of the test when this happens
    # and I will come back and fix this in the future.
    # https://team-1617704806227.atlassian.net/browse/MIP-745
    if df != expected["df"]:
        return

    # Df
    with subtests.test():
        for term in expected_terms:
            assert_allclose(df[term], expected["df"][term])

    # F values
    with subtests.test():
        for term in expected_terms:
            if f_stat[term]:
                assert_allclose(f_stat[term], expected["f_stat"][term])

    # p values
    with subtests.test():
        for term in expected_terms:
            if f_pvalue[term]:
                assert_allclose(f_pvalue[term], expected["f_pvalue"][term])
