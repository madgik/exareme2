import copy
import functools
import json

import numpy as np
import pytest
import requests


def algorithm_request(algorithm: str, input: dict):
    url = "http://127.0.0.1:5000/algorithms" + f"/{algorithm}"

    variables = copy.deepcopy(input["inputdata"]["y"])
    keys = input["inputdata"].keys()
    if "x" in keys and input["inputdata"]["x"]:
        variables.extend(input["inputdata"]["x"])

    filters = {
        "condition": "AND",
        "rules": [
            {
                "id": "dataset",
                "type": "string",
                "value": input["inputdata"]["datasets"],
                "operator": "in",
            },
        ],
        "valid": True,
    }
    input["inputdata"]["filters"] = filters
    request_json = json.dumps(input)

    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(url, data=request_json, headers=headers)
    if response.status_code != 200:
        raise ValueError(
            f"Unexpected response status: '{response.status_code}'. Response message: '{response.content}'"
        )
    try:
        json.loads(response.content)
    except json.decoder.JSONDecodeError as exc:
        print(response)
        raise exc
    return response


def get_test_params(expected_file, slc=None, skip_indices=None, skip_reason=None):
    """
    Gets parameters for algorithm validation tests from expected file

    Can get the whole list present in the expected file or a given slice. Can
    also skip some tests based on their indices.

    Parameters
    ----------
    expected_file : pathlib.Path
        File in json format containing a list of test cases, where a test case
        is a pair of input/output for a given algorithm
    slc : slice | None
        If not None it gets only the given slice
    skip_indices : list[int] | None
        Indices of tests to skip
    skip_reason : str | None
        Reason for skipping tests, combine with previous parameter
    """
    with expected_file.open() as f:
        params = json.load(f)["test_cases"]
    slc = slc or slice(len(params))
    params = [(p["input"], p["output"]) for p in params[slc]]

    def skip(*param):
        return pytest.param(*param, marks=pytest.mark.skip(reason=skip_reason))

    if skip_indices:
        params = [skip(*p) if i in skip_indices else p for i, p in enumerate(params)]
    return params


assert_allclose = functools.partial(
    np.testing.assert_allclose,
    rtol=1e-6,
    atol=1e-9,
)
