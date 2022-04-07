import json

import pytest
import requests

from tests.prod_env_tests import datasets_location_url


@pytest.fixture
def expected_datasets_location():
    return {
        "dementia:0.1": {
            "desd-synthdata0": ["localnode1"],
            "desd-synthdata1": ["localnode2"],
            "desd-synthdata2": ["localnode2"],
            "desd-synthdata3": ["localnode2"],
            "desd-synthdata4": ["localnode2"],
            "desd-synthdata5": ["localnode2"],
            "desd-synthdata6": ["localnode2"],
            "desd-synthdata7": ["localnode2"],
            "desd-synthdata8": ["localnode2"],
            "desd-synthdata9": ["localnode2"],
            "edsd0": ["localnode1"],
            "edsd1": ["localnode2"],
            "edsd2": ["localnode2"],
            "edsd3": ["localnode2"],
            "edsd4": ["localnode2"],
            "edsd5": ["localnode2"],
            "edsd6": ["localnode2"],
            "edsd7": ["localnode2"],
            "edsd8": ["localnode2"],
            "edsd9": ["localnode2"],
            "ppmi0": ["localnode1"],
            "ppmi1": ["localnode2"],
            "ppmi2": ["localnode2"],
            "ppmi3": ["localnode2"],
            "ppmi4": ["localnode2"],
            "ppmi5": ["localnode2"],
            "ppmi6": ["localnode2"],
            "ppmi7": ["localnode2"],
            "ppmi8": ["localnode2"],
            "ppmi9": ["localnode2"],
        },
        "tbi:0.1": {
            "dummy_tbi0": ["localnode1"],
            "dummy_tbi1": ["localnode2"],
            "dummy_tbi2": ["localnode2"],
            "dummy_tbi3": ["localnode2"],
            "dummy_tbi4": ["localnode2"],
            "dummy_tbi5": ["localnode2"],
            "dummy_tbi6": ["localnode2"],
            "dummy_tbi7": ["localnode2"],
            "dummy_tbi8": ["localnode2"],
            "dummy_tbi9": ["localnode2"],
        },
    }


def test_get_datasets_location(expected_datasets_location):
    request = requests.get(datasets_location_url)
    response = json.loads(request.text)
    for data_model in response:
        assert data_model in expected_datasets_location
        for dataset in response[data_model]:
            assert dataset in expected_datasets_location[data_model]
