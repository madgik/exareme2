import json

import pytest
import requests

from tests.prod_env_tests import datasets_url


@pytest.fixture
def expected_datasets_per_pathology():
    return {
        "dementia": {
            "edsd",
            "ppmi",
            "desd-synthdata",
        },
        "tbi": {"dummy_tbi"},
    }


def test_get_datasets(expected_datasets_per_pathology):
    request = requests.get(datasets_url)
    node_schemas = json.loads(request.text)

    datasets_per_pathology = {}
    for node_schema in node_schemas.values():
        for pathology, datasets in node_schema.items():
            if pathology not in datasets_per_pathology.keys():
                datasets_per_pathology[pathology] = set(datasets)
            else:
                datasets_per_pathology[pathology].update(datasets)

    assert datasets_per_pathology == expected_datasets_per_pathology
