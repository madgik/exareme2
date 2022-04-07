import json

import pytest
import requests

from tests.prod_env_tests import cdes_metadata_url


@pytest.fixture
def expected_data_model_len_metadata():
    return {
        "dementia:0.1": 186,
        "tbi:0.1": 21,
    }


def test_get_cdes_metadata(expected_data_model_len_metadata):
    request = requests.get(cdes_metadata_url)
    response = json.loads(request.text)

    for data_model, cdes in response.items():
        assert data_model in expected_data_model_len_metadata
        assert len(cdes["values"]) == expected_data_model_len_metadata[data_model]
