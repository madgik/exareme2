import json

import requests

from tests.prod_env_tests import datasets_url

expected_node_data_models = {
    "localnode1": {
        "dementia": [
            "edsd",
            "ppmi",
            "desd-synthdata",
        ],
        "tbi": ["dummy_tbi"],
    },
    "localnode2": {
        "dementia": [
            "edsd",
            "ppmi",
            "desd-synthdata",
        ],
        "tbi": ["dummy_tbi"],
    },
}


def test_get_datasets():
    request = requests.get(datasets_url)
    node_data_models = json.loads(request.text)
    assert len(node_data_models) == len(expected_node_data_models)
    assert set(node_data_models.keys()) == set(expected_node_data_models.keys())
    for node_id in node_data_models.keys():
        assert set(node_data_models[node_id].keys()) == set(
            expected_node_data_models[node_id].keys()
        )
        for data_model in node_data_models[node_id].keys():
            assert set(node_data_models[node_id][data_model]) == set(
                expected_node_data_models[node_id][data_model]
            )
