import json

import requests

from tests.e2e_tests import datasets_url

proper_node_schemas = {
    "worker1": {
        "dementia": [
            "edsd",
            "fake_longitudinal",
            "demo_data",
            "ppmi",
            "desd-synthdata",
        ],
        "mentalhealth": ["demo"],
        "tbi": ["dummy_tbi"],
    },
    "worker2": {
        "dementia": [
            "edsd",
            "fake_longitudinal",
            "demo_data",
            "ppmi",
            "desd-synthdata",
        ],
        "mentalhealth": ["demo"],
        "tbi": ["dummy_tbi"],
    },
}


def test_get_datasets():
    request = requests.get(datasets_url)
    node_schemas = json.loads(request.text)
    assert len(node_schemas) == len(proper_node_schemas)
    assert set(node_schemas.keys()) == set(proper_node_schemas.keys())
    for node_id in node_schemas.keys():
        assert set(node_schemas[node_id].keys()) == set(
            proper_node_schemas[node_id].keys()
        )
        for schema in node_schemas[node_id].keys():
            assert set(node_schemas[node_id][schema]) == set(
                proper_node_schemas[node_id][schema]
            )
