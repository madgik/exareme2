import json

import pytest
import requests

from tests.prod_env_tests import datasets_url


@pytest.fixture
def expected_datasets_per_data_model():
    return {
        "dementia:0.1": {
            "edsd0",
            "edsd1",
            "edsd2",
            "edsd3",
            "edsd4",
            "edsd5",
            "edsd6",
            "edsd7",
            "edsd8",
            "edsd9",
            "ppmi0",
            "ppmi1",
            "ppmi2",
            "ppmi3",
            "ppmi4",
            "ppmi5",
            "ppmi6",
            "ppmi7",
            "ppmi8",
            "ppmi9",
            "desd-synthdata0",
            "desd-synthdata1",
            "desd-synthdata2",
            "desd-synthdata3",
            "desd-synthdata4",
            "desd-synthdata5",
            "desd-synthdata6",
            "desd-synthdata7",
            "desd-synthdata8",
            "desd-synthdata9",
        },
        "tbi:0.1": {
            "dummy_tbi0",
            "dummy_tbi1",
            "dummy_tbi2",
            "dummy_tbi3",
            "dummy_tbi4",
            "dummy_tbi5",
            "dummy_tbi6",
            "dummy_tbi7",
            "dummy_tbi8",
            "dummy_tbi9",
        },
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
