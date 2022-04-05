import json

import pytest
import requests

from tests.prod_env_tests import dataset_locations_url


@pytest.fixture
def expected_dataset_locations():
    return {
        "dementia:0.1": {
            "desd-synthdata": ["localnode1", "localnode2"],
            "edsd": ["localnode1", "localnode2"],
            "ppmi": ["localnode1", "localnode2"],
        },
        "tbi:0.1": {"dummy_tbi": ["localnode1", "localnode2"]},
    }


def test_get_dataset_locations(expected_dataset_locations):
    request = requests.get(dataset_locations_url)
    response = json.loads(request.text)
    for data_model in response:
        assert data_model in expected_dataset_locations
        for dataset in response[data_model]:
            assert dataset in expected_dataset_locations[data_model]
        # TODO: Once dataset are present only on one node we will need to check their location too
