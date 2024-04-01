import json

import pytest
import requests

from tests.prod_env_tests import datasets_locations_url


@pytest.fixture
def expected_datasets_locations():
    return {
        "longitudinal_dementia:0.1": {
            "longitudinal_dementia0": "localworker1",
            "longitudinal_dementia1": "localworker2",
            "longitudinal_dementia2": "localworker3",
        },
        "tbi:0.1": {
            "dummy_tbi0": "localworker1",
            "dummy_tbi1": "localworker2",
            "dummy_tbi3": "localworker2",
            "dummy_tbi5": "localworker2",
            "dummy_tbi7": "localworker2",
            "dummy_tbi9": "localworker2",
            "dummy_tbi2": "localworker3",
            "dummy_tbi4": "localworker3",
            "dummy_tbi6": "localworker3",
            "dummy_tbi8": "localworker3",
        },
        "dementia:0.1": {
            "desd-synthdata0": "localworker1",
            "edsd0": "localworker1",
            "ppmi0": "localworker1",
            "desd-synthdata1": "localworker2",
            "desd-synthdata3": "localworker2",
            "desd-synthdata5": "localworker2",
            "desd-synthdata7": "localworker2",
            "desd-synthdata9": "localworker2",
            "edsd2": "localworker2",
            "edsd4": "localworker2",
            "edsd6": "localworker2",
            "edsd8": "localworker2",
            "ppmi1": "localworker2",
            "ppmi3": "localworker2",
            "ppmi5": "localworker2",
            "ppmi7": "localworker2",
            "ppmi9": "localworker2",
            "desd-synthdata2": "localworker3",
            "desd-synthdata4": "localworker3",
            "desd-synthdata6": "localworker3",
            "desd-synthdata8": "localworker3",
            "edsd1": "localworker3",
            "edsd3": "localworker3",
            "edsd5": "localworker3",
            "edsd7": "localworker3",
            "edsd9": "localworker3",
            "ppmi2": "localworker3",
            "ppmi4": "localworker3",
            "ppmi6": "localworker3",
            "ppmi8": "localworker3",
        },
    }


def test_get_dataset_location(expected_datasets_locations):
    request = requests.get(datasets_locations_url)
    response = json.loads(request.text)
    for worker_id in response:
        assert worker_id in expected_datasets_locations
        for data_model in response[worker_id]:
            assert data_model in expected_datasets_locations[worker_id]
            for dataset in response[worker_id][data_model]:
                assert dataset in response[worker_id][data_model]
