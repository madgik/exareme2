import copy
import functools
import json

import numpy as np
import requests


def algorithm_request(algorithm: str, input: dict):
    url = "http://127.0.0.1:5000/algorithms" + f"/{algorithm}"

    variables = copy.deepcopy(input["inputdata"]["y"])
    keys = input["inputdata"].keys()
    if "x" in keys:
        variables.extend(input["inputdata"]["x"])
    else:
        pass

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
                    for variable in variables
                ],
            },
        ],
        "valid": True,
    }
    input["inputdata"]["filters"] = filters
    request_json = json.dumps(input)

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(url, data=request_json, headers=headers)
    return response


def get_test_params(expected_file, slc=None):
    with expected_file.open() as f:
        params = json.load(f)["test_cases"]
    if not slc:
        slc = slice(len(params))
    params = [(p["input"], p["output"]) for p in params[slc]]
    return params


assert_allclose = functools.partial(
    np.testing.assert_allclose,
    rtol=1e-6,
    atol=1e-9,
)
