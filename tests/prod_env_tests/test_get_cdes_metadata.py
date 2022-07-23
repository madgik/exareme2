import json

import requests

from tests.prod_env_tests import cdes_metadata_url


def test_get_cdes_metadata():
    request = requests.get(cdes_metadata_url)
    response = json.loads(request.text)

    for data_model, cdes in response.items():
        assert data_model in ["dementia:0.1", "tbi:0.1"]
        assert len(cdes) > 0
        assert "dataset" in cdes
