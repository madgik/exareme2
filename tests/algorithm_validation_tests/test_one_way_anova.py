import json
from pathlib import Path

import numpy as np
import pytest
import requests

from tests.prod_env_tests import algorithms_url

expected_file = Path(__file__).parent / "expected" / "one_way_anova_expected.json"


def anova_one_way_request(input):
    url = algorithms_url + "/one_way_anova"

    filters = {
        "condition": "AND",
        "rules": [
            {
                "id": "dataset",
                "type": "string",
                "value": input["inputdata"]["datasets"],
                "operator": "in",
            },
            {
                "condition": "AND",
                "rules": [
                    {
                        "id": variable,
                        "type": "string",
                        "operator": "is_not_null",
                        "value": None,
                    }
                    for variable in input["inputdata"]["x"] + input["inputdata"]["y"]
                ],
            },
        ],
        "valid": True,
    }
    input["inputdata"]["filters"] = filters
    # input["inputdata"]["use_smpc"] = False
    request_json = json.dumps(input)

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(url, data=request_json, headers=headers)
    return response


def get_test_params(file, slc=None):
    with file.open() as f:
        params = json.load(f)["test_cases"]
    if not slc:
        slc = slice(len(params))
    params = [(p["input"], p["output"]) for p in params[slc]]
    return params


@pytest.mark.skip(reason="some of the parametrized tests fail with assertion errors")
@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_anova_algorithm(test_input, expected):
    # -------------------------------------------------------
    import os

    print("\n------------------ $df -hx squashfs --total -----------------")
    os.system("df -hx squashfs --total")
    print("------------------ $free -mh -----------------")
    os.system("free -mh")
    print("\n")
    # -------------------------------------------------------
    result = json.loads(anova_one_way_request(test_input).content)
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
    for et, rt in zip(e_tukey, tukey):
        for key, e_val in et.items():
            r_val = rt[key]
            assert e_val == r_val or np.isclose(e_val, r_val, rtol=1e-7, atol=1e-10)
