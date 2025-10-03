import functools
import json

import numpy as np
import pytest
import requests


def algorithm_request(algorithm: str, input: dict):
    url = "http://127.0.0.1:5000/algorithms" + f"/{algorithm}"
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(url, data=json.dumps(input), headers=headers)
    return response


def parse_response(response) -> dict:
    if response.status_code != 200:
        msg = f"Unexpected response status: '{response.status_code}'. "
        msg += f"Response message: '{response.content}'"
        raise ValueError(msg)
    try:
        result = json.loads(response.content)
    except json.decoder.JSONDecodeError:
        raise ValueError(f"The result is not valid json:\n{response.content}") from None
    return result
