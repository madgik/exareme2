import json

import requests

from tests.dev_env_tests import algorithms_url


def test_bad_filters_exception():
    algorithm_name = "standard_deviation"
    request_params = {
        "inputdata": {
            "data_model": "dementia:0.1",
            "datasets": ["edsd0"],
            "x": [
                "lefthippocampus",
            ],
            "filters": {"whateveeeeeer": "!!!"},
        },
    }

    algorithm_url = algorithms_url + "/" + algorithm_name
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        algorithm_url,
        data=json.dumps(request_params),
        headers=headers,
    )

    assert "Invalid filters format." in response.text
    assert response.status_code == 400
