import json

import pytest
import requests

from tests.prod_env_tests import datasets_locations_url


@pytest.fixture
def expected_datasets_locations():
    return {
        "longitudinal_dementia:0.1": {
            "longitudinal_dementia0": "localnode1",
            "longitudinal_dementia1": "localnode2",
            "longitudinal_dementia2": "localnode3",
        },
        "tbi:0.1": {
            "dummy_tbi0": "localnode1",
            "dummy_tbi1": "localnode2",
            "dummy_tbi3": "localnode2",
            "dummy_tbi5": "localnode2",
            "dummy_tbi7": "localnode2",
            "dummy_tbi9": "localnode2",
            "dummy_tbi2": "localnode3",
            "dummy_tbi4": "localnode3",
            "dummy_tbi6": "localnode3",
            "dummy_tbi8": "localnode3",
        },
        "dementia:0.1": {
            "desd-synthdata0": "localnode1",
            "edsd0": "localnode1",
            "ppmi0": "localnode1",
            "desd-synthdata1": "localnode2",
            "desd-synthdata3": "localnode2",
            "desd-synthdata5": "localnode2",
            "desd-synthdata7": "localnode2",
            "desd-synthdata9": "localnode2",
            "edsd2": "localnode2",
            "edsd4": "localnode2",
            "edsd6": "localnode2",
            "edsd8": "localnode2",
            "ppmi1": "localnode2",
            "ppmi3": "localnode2",
            "ppmi5": "localnode2",
            "ppmi7": "localnode2",
            "ppmi9": "localnode2",
            "desd-synthdata2": "localnode3",
            "desd-synthdata4": "localnode3",
            "desd-synthdata6": "localnode3",
            "desd-synthdata8": "localnode3",
            "edsd1": "localnode3",
            "edsd3": "localnode3",
            "edsd5": "localnode3",
            "edsd7": "localnode3",
            "edsd9": "localnode3",
            "ppmi2": "localnode3",
            "ppmi4": "localnode3",
            "ppmi6": "localnode3",
            "ppmi8": "localnode3",
        },
    }


def test_get_dataset_location(expected_datasets_locations):
    request = requests.get(datasets_locations_url)
    response = json.loads(request.text)
    for node_id in response:
        assert node_id in expected_datasets_locations
        for data_model in response[node_id]:
            assert data_model in expected_datasets_locations[node_id]
            for dataset in response[node_id][data_model]:
                assert dataset in response[node_id][data_model]
