import json
from pathlib import Path

import numpy as np
import pytest
import requests

expected_file = Path(__file__).parent / "expected" / "one_way_anova_expected.json"


def anova_one_way_request(input):
    url = "http://127.0.0.1:5000/algorithms" + "/one_way_anova"

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


@pytest.mark.parametrize("test_input, expected", get_test_params(expected_file))
def test_anova_algorithm_local(test_input, expected):
    result = json.loads(anova_one_way_request(test_input).content)
    aov = result["anova_table"]
    tukey = result["tukey_test"]
    e_aov = {k: v for k, v in expected.items() if k != "tukey_test"}
    e_tukey = expected["tukey_test"]
    assert set(e_aov) == set(aov.keys())
    for key, e_val in e_aov.items():
        r_val = aov[key]
        assert np.isclose(e_val, r_val)
    for et, rt in zip(e_tukey, tukey):
        for key, e_val in et.items():
            r_val = rt[key]
            assert e_val == r_val or np.isclose(e_val, r_val)
