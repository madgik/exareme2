import json
from unittest.mock import DEFAULT
from unittest.mock import patch

import pytest

from mipengine.controller.api.exceptions import BadRequest
from mipengine.controller.api.exceptions import BadUserInput
from mipengine.controller.api.validator import validate_algorithm_request
from tests.unit_tests.common_data_elements import (
    common_data_elements as mocked_common_data_elements,
)
from tests.unit_tests.algorithms_specifications import (
    algorithms_specifications as mocked_algorithms_specifications,
)


@pytest.fixture(scope="module", autouse=True)
def mock_cdes():
    with patch(
        "mipengine.controller.api.validator.common_data_elements",
        mocked_common_data_elements,
    ):
        yield


@pytest.fixture(scope="module", autouse=True)
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
                ("test_pathology1", "test_dataset2"),
                ("test_pathology2", "test_dataset2"),
                ("test_pathology2", "test_dataset3"),
            }
            else False
        )

        yield


@pytest.fixture(scope="module", autouse=True)
def mock_algorithms_specs():
    with patch(
        "mipengine.controller.api.validator.algorithms_specifications",
        mocked_algorithms_specifications,
    ):
        yield


test_cases_proper_validate_algorithm = [
    (
        "test_algorithm1",
        {
            "inputdata": {
                "pathology": "test_pathology1",
                "datasets": ["test_dataset1", "test_dataset2"],
                "x": ["test_cde1", "test_cde2"],
                "y": ["test_cde3"],
            },
            "parameters": {"parameter1": [1, 3], "parameter2": 3},
        },
    ),
    (
        "test_algorithm1",
        {
            "inputdata": {
                "pathology": "test_pathology2",
                "datasets": ["test_dataset2", "test_dataset3"],
                "filters": None,
                "x": ["test_cde1"],
                "y": ["test_cde3"],
            },
            "parameters": {"parameter1": [1, 3], "parameter2": 3},
        },
    ),
]


@pytest.mark.parametrize(
    "algorithm_name, request_body_dict", test_cases_proper_validate_algorithm
)
def test_validate_algorithm_success(
    algorithm_name,
    request_body_dict,
):
    validate_algorithm_request(algorithm_name, json.dumps(request_body_dict))


test_cases_validate_algorithm_exceptions = [
    ("non_existing_algorithm", None, (BadRequest, "Algorithm .* does not exist.")),
    (
        "test_algorithm1",
        {"wrong_dto": 3},
        (BadRequest, "The algorithm request body .*"),
    ),
    (
        "test_algorithm1",
        {
            "inputdata": {
                "pathology": "test_pathology1",
                "datasets": 123,
                "x": ["lefthippocampus", "righthippocampus"],
                "y": ["alzheimerbroadcategory_bin"],
            },
        },
        (BadRequest, "The algorithm request body .*"),
    ),
    (
        "test_algorithm1",
        {
            "inputdata": {
                "pathology": "test_pathology1",
                "datasets": ["demo_data"],
                "x": ["lefthippocampus", "righthippocampus"],
                "y": ["alzheimerbroadcategory_bin"],
            },
        },
        (BadUserInput, "Datasets .* do not belong in pathology .*"),
    ),
    (
        "test_algorithm1",
        {
            "inputdata": {
                "pathology": "non_existing",
                "datasets": ["demo_data"],
                "x": ["lefthippocampus", "righthippocampus"],
                "y": ["alzheimerbroadcategory_bin"],
            },
        },
        (BadUserInput, "Pathology .* does not exist."),
    ),
    (
        "test_algorithm1",
        {
            "inputdata": {
                "pathology": "test_pathology1",
                "datasets": ["non_existing", "non_existing2"],
                "x": ["lefthippocampus", "righthippocampus"],
                "y": ["alzheimerbroadcategory_bin"],
            },
        },
        (BadUserInput, "Datasets .* do not belong in pathology .*"),
    ),
    (
        "test_algorithm1",
        {
            "inputdata": {
                "pathology": "test_pathology1",
                "datasets": ["test_dataset1"],
                "y": ["test_cde3"],
            },
        },
        (BadUserInput, "Inputdata .* should be provided."),
    ),
    # TODO Should be enabled again when the dataclasses are replaced with pydantic
    # (
    #     "test_algorithm1",
    #     {
    #         "inputdata": {
    #             "pathology": "test_pathology1",
    #             "datasets": ["test_dataset1"],
    #             "x": "test_cde1",
    #             "y": ["test_cde3"],
    #         },
    #     },
    #     (BadUserInput, "Inputdata .* should be a list."),
    # ),
    (
        "test_algorithm1",
        {
            "inputdata": {
                "pathology": "test_pathology1",
                "datasets": ["test_dataset1"],
                "x": ["test_cde1", "test_cde2"],
                "y": ["test_cde3", "test_cde2"],
            },
        },
        (BadUserInput, "Inputdata .* cannot have multiple values."),
    ),
    (
        "test_algorithm1",
        {
            "inputdata": {
                "pathology": "test_pathology1",
                "datasets": ["test_dataset1"],
                "x": ["test_cde1", "test_cde2"],
                "y": ["non_existing"],
            },
        },
        (BadUserInput, "The CDE .* does not exist in pathology .*"),
    ),
    (
        "test_algorithm1",
        {
            "inputdata": {
                "pathology": "test_pathology1",
                "datasets": ["test_dataset1"],
                "x": ["test_cde1", "test_cde2"],
                "y": ["test_cde1"],
            },
        },
        (BadUserInput, "The CDE .* doesn't have one of the allowed types .*"),
    ),
    (
        "test_algorithm1",
        {
            "inputdata": {
                "pathology": "test_pathology1",
                "datasets": ["test_dataset1"],
                "x": ["test_cde1", "test_cde2"],
                "y": ["test_cde4"],
            },
        },
        (BadUserInput, "The CDE .* should be categorical."),
    ),
    (
        "test_algorithm1",
        {
            "inputdata": {
                "pathology": "test_pathology1",
                "datasets": ["test_dataset1"],
                "x": ["test_cde5", "test_cde2"],
                "y": ["test_cde3"],
            },
        },
        (BadUserInput, "The CDE .* should NOT be categorical."),
    ),
    (
        "test_algorithm1",
        {
            "inputdata": {
                "pathology": "test_pathology1",
                "datasets": ["test_dataset1"],
                "x": ["test_cde1", "test_cde2"],
                "y": ["test_cde6"],
            },
        },
        (BadUserInput, "The CDE .* should have .* enumerations."),
    ),
    (
        "test_algorithm1",
        {
            "inputdata": {
                "pathology": "test_pathology1",
                "datasets": ["test_dataset1"],
                "x": ["test_cde1", "test_cde2"],
                "y": ["test_cde3"],
            },
        },
        (BadUserInput, "Algorithm parameters not provided."),
    ),
    (
        "test_algorithm1",
        {
            "inputdata": {
                "pathology": "test_pathology1",
                "datasets": ["test_dataset1"],
                "x": ["test_cde1", "test_cde2"],
                "y": ["test_cde3"],
            },
            "parameters": {"wrong_parameter": ""},
        },
        (BadUserInput, "Parameter .* should not be blank."),
    ),
    (
        "test_algorithm1",
        {
            "inputdata": {
                "pathology": "test_pathology1",
                "datasets": ["test_dataset1"],
                "x": ["test_cde1", "test_cde2"],
                "y": ["test_cde3"],
            },
            "parameters": {"parameter1": 2},
        },
        (BadUserInput, "Parameter .* should be a list."),
    ),
    (
        "test_algorithm1",
        {
            "inputdata": {
                "pathology": "test_pathology1",
                "datasets": ["test_dataset1"],
                "x": ["test_cde1", "test_cde2"],
                "y": ["test_cde3"],
            },
            "parameters": {"parameter1": [1, 4]},
        },
        (BadUserInput, "Parameter .* values should be one of the following: .*"),
    ),
    (
        "test_algorithm1",
        {
            "inputdata": {
                "pathology": "test_pathology1",
                "datasets": ["test_dataset1"],
                "x": ["test_cde1", "test_cde2"],
                "y": ["test_cde3"],
            },
            "parameters": {"parameter1": [1], "parameter2": 1},
        },
        (BadUserInput, "Parameter .* values should be greater than .*"),
    ),
    (
        "test_algorithm1",
        {
            "inputdata": {
                "pathology": "test_pathology1",
                "datasets": ["test_dataset1"],
                "x": ["test_cde1", "test_cde2"],
                "y": ["test_cde3"],
            },
            "parameters": {"parameter1": [1], "parameter2": 10},
        },
        (BadUserInput, "Parameter .* values should be less than .*"),
    ),
]


@pytest.mark.parametrize(
    "algorithm_name, request_body_dict, exception",
    test_cases_validate_algorithm_exceptions,
)
def test_validate_algorithm_exceptions(algorithm_name, request_body_dict, exception):
    exception_type, exception_message = exception
    with pytest.raises(exception_type, match=exception_message):
        validate_algorithm_request(algorithm_name, json.dumps(request_body_dict))
