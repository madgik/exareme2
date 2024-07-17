import json

import pytest
import requests

from tests.prod_env_tests import datasets_locations_url


@pytest.fixture
def expected_datasets_locations():
    return {
        "dementia:0.1": {
            "desd-synthdata0": "localworker2",
            "desd-synthdata1": "localworker1",
            "desd-synthdata2": "localworker2",
            "desd-synthdata3": "localworker1",
            "desd-synthdata4": "localworker2",
            "desd-synthdata5": "localworker1",
            "desd-synthdata6": "localworker2",
            "desd-synthdata7": "localworker1",
            "desd-synthdata8": "localworker2",
            "desd-synthdata9": "localworker1",
            "desd-synthdata_test": "globalworker",
            "edsd0": "localworker2",
            "edsd1": "localworker1",
            "edsd2": "localworker2",
            "edsd3": "localworker1",
            "edsd4": "localworker2",
            "edsd5": "localworker1",
            "edsd6": "localworker2",
            "edsd7": "localworker1",
            "edsd8": "localworker2",
            "edsd9": "localworker1",
            "edsd_test": "globalworker",
            "ppmi0": "localworker2",
            "ppmi1": "localworker1",
            "ppmi2": "localworker2",
            "ppmi3": "localworker1",
            "ppmi4": "localworker2",
            "ppmi5": "localworker1",
            "ppmi6": "localworker2",
            "ppmi7": "localworker1",
            "ppmi8": "localworker2",
            "ppmi9": "localworker1",
            "ppmi_test": "globalworker",
        },
        "longitudinal_dementia:0.1": {
            "longitudinal_dementia0": "localworker2",
            "longitudinal_dementia1": "localworker1",
            "longitudinal_dementia2": "localworker2",
            "longitudinal_dementia_test": "globalworker",
        },
        "tbi:0.1": {
            "dummy_tbi0": "localworker2",
            "dummy_tbi1": "localworker1",
            "dummy_tbi2": "localworker2",
            "dummy_tbi3": "localworker1",
            "dummy_tbi4": "localworker2",
            "dummy_tbi5": "localworker1",
            "dummy_tbi6": "localworker2",
            "dummy_tbi7": "localworker1",
            "dummy_tbi8": "localworker2",
            "dummy_tbi9": "localworker1",
            "dummy_tbi_test": "globalworker",
        },
    }


def test_get_dataset_location(expected_datasets_locations):
    request = requests.get(datasets_locations_url)
    response = json.loads(request.text)
    assert response == expected_datasets_locations
