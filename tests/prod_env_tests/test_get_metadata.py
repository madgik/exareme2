import json

import pytest
import requests

from mipengine.node_tasks_DTOs import CommonDataElements
from tests.prod_env_tests import metadata_url


@pytest.fixture
def expected_data_model_len_metadata():
    return {
        "dementia:0.1": 186,
        "tbi:0.1": 21,
    }


def test_get_metadata(expected_data_model_len_metadata):
    request = requests.get(metadata_url)
    response = json.loads(request.text)

    for data_model, cdes in response.items():
        assert data_model in expected_data_model_len_metadata
        assert len(cdes["values"]) == expected_data_model_len_metadata[data_model]
