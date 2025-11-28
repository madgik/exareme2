import json

import requests

from tests.prod_env_tests import data_models_attributes_url


def test_get_data_models_attributes():
    request = requests.get(data_models_attributes_url)
    response = json.loads(request.text)

    for data_model, attributes in response.items():
        assert data_model in ["dementia:0.1", "tbi:0.1", "longitudinal_dementia:0.1"]
        assert all([elem in attributes for elem in ["tags", "properties"]])
        assert len(attributes["properties"]) > 0
