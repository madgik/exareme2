import json
from unittest.mock import DEFAULT
from unittest.mock import patch

import pytest

from mipengine.controller.api.exceptions import BadRequest
from mipengine.controller.api.exceptions import BadUserInput
from mipengine.controller.api.validate_algorithm import validate_algorithm
from tests.unit_tests.common_data_elements import (
    common_data_elements as mocked_common_data_elements,
)


# TODO Add algorithm_specifications mock


@pytest.fixture(scope="session", autouse=True)
def mock_cdes():
    with patch(
        "mipengine.controller.api.validate_algorithm.common_data_elements",
        mocked_common_data_elements,
    ):
        yield


@pytest.fixture(scope="session", autouse=True)
def mock_node_catalog():
    with patch.multiple(
        "mipengine.common.node_catalog.NodeCatalog",
        pathology_exists=DEFAULT,
        dataset_exists=DEFAULT,
    ) as mock_node_catalog:
        mock_node_catalog["pathology_exists"].side_effect = (
            lambda x: True
            if x
            in {
                "test_pathology1",
                "test_pathology2",
            }
            else False
        )

        mock_node_catalog["dataset_exists"].side_effect = (
            lambda *x: True
            if x
            in {
                ("test_pathology1", "test_dataset1"),
                ("test_pathology2", "test_dataset2"),
            }
            else False
        )

        yield


test_cases_proper_validate_algorithm = [
    (
        "logistic_regression",
        {
            "inputdata": {
                "pathology": "test_pathology1",
                "datasets": ["test_dataset1"],
                "filters": None,
                "x": ["test_cde1", "test_cde2"],
                "y": ["test_cde3"],
            },
            "parameters": None,
            "crossvalidation": None,
        },
    )
]


@pytest.mark.parametrize(
    "algorithm_name, request_body_dict", test_cases_proper_validate_algorithm
)
def test_validate_algorithm_success(
    algorithm_name,
    request_body_dict,
):
    validate_algorithm(algorithm_name, json.dumps(request_body_dict))


test_cases_validate_algorithm_exceptions = [
    ("non_existing_algorithm", None, BadRequest),
    ("logistic_regression", {"wrong_dto": 3}, BadRequest),
    (
        "logistic_regression",
        {
            "inputdata": {
                "pathology": "non_existing",
                "datasets": ["demo_data"],
                "filters": None,
                "x": ["lefthippocampus", "righthippocampus"],
                "y": ["alzheimerbroadcategory_bin"],
            },
            "parameters": None,
            "crossvalidation": None,
        },
        BadUserInput,
    ),
    (
        "logistic_regression",
        {
            "inputdata": {
                "pathology": "dementia",
                "datasets": 123,
                "filters": None,
                "x": ["lefthippocampus", "righthippocampus"],
                "y": ["alzheimerbroadcategory_bin"],
            },
            "parameters": None,
            "crossvalidation": None,
        },
        BadRequest,
    ),
    (
        "logistic_regression",
        {
            "inputdata": {
                "pathology": "dementia",
                "datasets": ["non_existing"],
                "filters": None,
                "x": ["lefthippocampus", "righthippocampus"],
                "y": ["alzheimerbroadcategory_bin"],
            },
            "parameters": None,
            "crossvalidation": None,
        },
        BadUserInput,
    ),
]


@pytest.mark.parametrize(
    "algorithm_name, request_body_dict, exception_type",
    test_cases_validate_algorithm_exceptions,
)
def test_validate_algorithm_exceptions(
    algorithm_name, request_body_dict, exception_type
):
    with pytest.raises(exception_type):
        from mipengine.controller.api.validate_algorithm import validate_algorithm

        validate_algorithm(algorithm_name, json.dumps(request_body_dict))
