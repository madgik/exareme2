import json

import pytest
import requests

from tests.prod_env_tests import datasets_url


@pytest.fixture
def expected_datasets_per_data_model():
    return {
        "dementia:0.1": {
            "edsd",
            "ppmi",
            "desd-synthdata",
        },
        "tbi:0.1": {"dummy_tbi"},
    }


def test_get_datasets(expected_datasets_per_data_model):
    request = requests.get(datasets_url)
    response = json.loads(request.text)
    datasets_per_data_model = {}
    for data_model, datasets in response.items():
        if data_model not in datasets_per_data_model.keys():
            datasets_per_data_model[data_model] = set(datasets)
        else:
            datasets_per_data_model[data_model].update(datasets)

    assert datasets_per_data_model == expected_datasets_per_data_model
