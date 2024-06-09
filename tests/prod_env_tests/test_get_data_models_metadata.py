import json

import requests

from tests.prod_env_tests import data_models_metadata_url


def test_get_data_models_metadata():
    request = requests.get(data_models_metadata_url)
    response = json.loads(request.text)

    for data_model, metadata in response.items():
        assert data_model in ["dementia:0.1", "tbi:0.1", "longitudinal_dementia:0.1"]
        assert all([elem in metadata for elem in ["tags", "properties"]])
        assert "cdes" in metadata["properties"]
        assert len(metadata["properties"]["cdes"]) > 0
