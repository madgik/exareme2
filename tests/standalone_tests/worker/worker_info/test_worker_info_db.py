import json

import pytest
from pydantic import ValidationError

from exareme2.worker_communication import DatasetProperties


def test_dataset_properties_parse_raw_returns_variables():
    payload = {
        "tags": [],
        "properties": {"variables": ["col1", "col2"]},
    }

    result = DatasetProperties.parse_raw(json.dumps(payload))

    assert result.variables == ["col1", "col2"]


def test_dataset_properties_missing_variables_key_raises_validation_error():
    payload = {"tags": [], "properties": {}}

    with pytest.raises(ValidationError):
        DatasetProperties.parse_raw(json.dumps(payload))


def test_dataset_properties_variables_not_list_raises_validation_error():
    payload = {"tags": [], "properties": {"variables": "not-a-list"}}

    with pytest.raises(ValidationError):
        DatasetProperties.parse_raw(json.dumps(payload))
