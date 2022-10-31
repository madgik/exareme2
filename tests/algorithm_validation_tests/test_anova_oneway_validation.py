import json
from pathlib import Path

import numpy as np
import pytest

from tests.algorithm_validation_tests.helpers import algorithm_request

expected_file = Path(__file__).parent / "expected" / "anova_oneway_expected.json"


def get_test_params(file, slc=None):
    with file.open() as f:
        params = json.load(f)["test_cases"]
    if not slc:
        slc = slice(len(params))
    params = [(p["input"], p["output"]) for p in params[slc]]
    return params


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_anova_algorithm(test_input, expected):
    response = algorithm_request("anova_oneway", test_input)

    if response.status_code != 200:
        raise ValueError(
            f"Unexpected response status: '{response.status_code}'. Response message: '{response.content}'"
        )
    result = json.loads(response.content)
    aov = result["anova_table"]
    tukey = result["tuckey_test"]

    e_aov = {k: v for k, v in expected.items() if k != "tuckey_test"}
    e_tukey = expected["tuckey_test"]
    res_aov = set(aov.keys())
    res_aov.remove("x_label")
    res_aov.remove("y_label")

    assert set(e_aov) == res_aov
    for key, e_val in e_aov.items():
        r_val = aov[key]
        assert e_val == r_val or np.isclose(e_val, r_val)
    compare_results(e_tukey, tukey)


def compare_results(expected, result):
    assert len(expected) == len(result)
    for grp_res_a in expected:
        groupA_exp = grp_res_a["groupA"]
        groupB_exp = grp_res_a["groupB"]
        for grp_res_b in result:
            groupA_res = grp_res_b["groupA"]
            groupB_res = grp_res_b["groupB"]
            if groupA_exp == groupA_res and groupB_exp == groupB_res:
                compare_as_is(grp_res_a, grp_res_b)
                break
            elif groupA_exp == groupB_res and groupB_exp == groupA_res:
                compare_opposites(grp_res_a, grp_res_b)
                break
        else:
            raise Exception(
                f"Group categories {groupA_exp} - {groupB_exp} could not be found in {result}"
            )


def compare_as_is(expected, result):
    for key, e_val in expected.items():
        r_val = result[key]
        if isinstance(e_val, str) and isinstance(r_val, str):
            assert e_val == r_val
        else:
            assert np.isclose(e_val, r_val, rtol=1e-7, atol=1e-10)


def compare_opposites(expected, result):
    assert expected["groupA"] == result["groupB"]
    assert expected["groupB"] == result["groupA"]
    assert np.isclose(expected["meanA"], result["meanB"], rtol=1e-7, atol=1e-10)
    assert np.isclose(expected["meanB"], result["meanA"], rtol=1e-7, atol=1e-10)
    assert np.isclose(expected["diff"], result["diff"] * -1, rtol=1e-7, atol=1e-10)
    assert np.isclose(expected["se"], result["se"], rtol=1e-7, atol=1e-10)
    assert np.isclose(expected["t_stat"], result["t_stat"] * -1, rtol=1e-7, atol=1e-10)
    assert np.isclose(expected["p_tuckey"], result["p_tuckey"], rtol=1e-7, atol=1e-10)
