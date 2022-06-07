from unittest.mock import patch

import pytest

from mipengine.controller.algorithm_specifications import AlgorithmSpecification
from mipengine.controller.algorithm_specifications import AlgorithmSpecifications
from mipengine.controller.algorithm_specifications import InputDataSpecification
from mipengine.controller.algorithm_specifications import InputDataSpecifications
from mipengine.controller.algorithm_specifications import ParameterSpecification
from mipengine.controller.api.algorithm_request_dto import AlgorithmInputDataDTO
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO
from mipengine.controller.api.exceptions import BadRequest
from mipengine.controller.api.exceptions import BadUserInput
from mipengine.controller.api.validator import validate_algorithm_request
from mipengine.controller.node_landscape_aggregator import NodeLandscapeAggregator
from mipengine.node_tasks_DTOs import CommonDataElement
from mipengine.node_tasks_DTOs import CommonDataElements


@pytest.fixture(scope="module", autouse=True)
def mock_cdes():
    node_landscape_aggregator = NodeLandscapeAggregator()
    data_models = {
        "test_data_model1:0.1": CommonDataElements(
            values={
                "test_cde1": CommonDataElement(
                    code="test_cde1",
                    label="test cde1",
                    sql_type="int",
                    is_categorical=False,
                    enumerations=None,
                    min=None,
                    max=None,
                ),
                "test_cde2": CommonDataElement(
                    code="test_cde2",
                    label="test cde2",
                    sql_type="real",
                    is_categorical=False,
                    enumerations=None,
                    min=None,
                    max=None,
                ),
                "test_cde3": CommonDataElement(
                    code="test_cde3",
                    label="test cde3",
                    sql_type="text",
                    is_categorical=True,
                    enumerations={"male": "male", "female": "female"},
                    min=None,
                    max=None,
                ),
                "test_cde4": CommonDataElement(
                    code="test_cde4",
                    label="test cde4",
                    sql_type="text",
                    is_categorical=False,
                    min=None,
                    max=None,
                ),
                "test_cde5": CommonDataElement(
                    code="test_cde5",
                    label="test cde5",
                    sql_type="int",
                    is_categorical=True,
                    enumerations={
                        "1": "1",
                        "2": "2",
                    },
                    min=None,
                    max=None,
                ),
                "test_cde6": CommonDataElement(
                    code="test_cde6",
                    label="test cde6",
                    sql_type="text",
                    is_categorical=True,
                    enumerations={"male": "male", "female": "female", "Other": "Other"},
                    min=None,
                    max=None,
                ),
            }
        ),
        "test_data_model2:0.1": CommonDataElements(
            values={
                "test_cde1": CommonDataElement(
                    code="test_cde1",
                    label="test cde1",
                    sql_type="int",
                    is_categorical=False,
                    enumerations=None,
                    min=None,
                    max=None,
                ),
                "test_cde2": CommonDataElement(
                    code="test_cde2",
                    label="test cde2",
                    sql_type="real",
                    is_categorical=False,
                    enumerations=None,
                    min=None,
                    max=None,
                ),
                "test_cde3": CommonDataElement(
                    code="test_cde3",
                    label="test cde3",
                    sql_type="text",
                    is_categorical=True,
                    enumerations={"male": "male", "female": "female"},
                    min=None,
                    max=None,
                ),
            }
        ),
    }

    node_landscape_aggregator._data_model_registry.data_models = data_models

    with patch(
        "mipengine.controller.api.validator.node_landscape_aggregator",
        node_landscape_aggregator,
    ):
        yield


@pytest.fixture()
def available_datasets_per_data_model():
    d = {
        "test_data_model1:0.1": ["test_dataset1", "test_dataset2"],
        "test_data_model2:0.1": ["test_dataset2", "test_dataset3"],
    }
    return d


@pytest.fixture(scope="module", autouse=True)
def mock_algorithms_specs():
    algorithms_specifications = AlgorithmSpecifications()
    algorithms_specifications.enabled_algorithms = {
        "test_algorithm1": AlgorithmSpecification(
            name="test algorithm1",
            desc="test algorithm1",
            label="test algorithm1",
            enabled=True,
            inputdata=InputDataSpecifications(
                x=InputDataSpecification(
                    label="features",
                    desc="Features",
                    types=["real"],
                    stattypes=["numerical"],
                    notblank=True,
                    multiple=True,
                    enumslen=None,
                ),
                y=InputDataSpecification(
                    label="target",
                    desc="Target variable",
                    types=["text"],
                    stattypes=["nominal"],
                    notblank=True,
                    multiple=False,
                    enumslen=2,
                ),
            ),
            parameters={
                "parameter1": ParameterSpecification(
                    label="paremeter1",
                    desc="parameter 1",
                    type="real",
                    notblank=True,
                    multiple=True,
                    default=1,
                    enums=[1, 2, 3],
                ),
                "parameter2": ParameterSpecification(
                    label="paremeter2",
                    desc="parameter 2",
                    type="int",
                    notblank=False,
                    multiple=False,
                    default=None,
                    min=2,
                    max=5,
                ),
                "parameter3": ParameterSpecification(
                    label="paremeter3",
                    desc="parameter 3",
                    type="text",
                    notblank=False,
                    multiple=False,
                    default=None,
                ),
                "parameter4": ParameterSpecification(
                    label="paremeter4",
                    desc="parameter 4",
                    type="int",
                    notblank=False,
                    multiple=True,
                    default=1,
                ),
            },
            flags={"formula": False},
        ),
        "algorithm_without_x": AlgorithmSpecification(
            name="algorithm_without_x",
            desc="algorithm_without_x",
            label="algorithm_without_x",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="features",
                    desc="Features",
                    types=["real"],
                    stattypes=["numerical"],
                    notblank=True,
                    multiple=True,
                    enumslen=None,
                ),
            ),
        ),
    }

    with patch(
        "mipengine.controller.api.validator.algorithm_specifications",
        algorithms_specifications,
    ):
        yield


def get_parametrization_list_success_cases():
    parametrization_list = [
        (
            "test_algorithm1",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model1:0.1",
                    datasets=["test_dataset1", "test_dataset2"],
                    x=["test_cde1", "test_cde2"],
                    y=["test_cde3"],
                ),
                parameters={"parameter1": [1, 3], "parameter2": 3},
            ),
        ),
        (
            "test_algorithm1",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model2:0.1",
                    datasets=["test_dataset2", "test_dataset3"],
                    x=["test_cde1"],
                    y=["test_cde3"],
                ),
                parameters={"parameter1": [1, 3], "parameter2": 3},
            ),
        ),
        (
            "algorithm_without_x",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model2:0.1",
                    datasets=["test_dataset2", "test_dataset3"],
                    y=["test_cde1"],
                ),
            ),
        ),
    ]
    return parametrization_list


@pytest.mark.parametrize(
    "algorithm_name, request_dto", get_parametrization_list_success_cases()
)
def test_validate_algorithm_success(
    algorithm_name, request_dto, available_datasets_per_data_model
):
    validate_algorithm_request(
        algorithm_name=algorithm_name,
        algorithm_request_dto=request_dto,
        available_datasets_per_data_model=available_datasets_per_data_model,
    )


def get_parametrization_list_exception_cases():
    parametrization_list = [
        (
            "non_existing_algorithm",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model1:0.1",
                    datasets=["test_data"],
                    x=["lefthippocampus", "righthippocampus"],
                    y=["alzheimerbroadcategory_bin"],
                )
            ),
            (BadRequest, "Algorithm .* does not exist."),
        ),
        (
            "test_algorithm1",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model1:0.1",
                    datasets=["test_data"],
                    x=["lefthippocampus", "righthippocampus"],
                    y=["alzheimerbroadcategory_bin"],
                )
            ),
            (
                BadUserInput,
                "Datasets:.* could not be found for data_model:.*",
            ),
        ),
        (
            "test_algorithm1",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="non_existing",
                    datasets=["test_data"],
                    x=["lefthippocampus", "righthippocampus"],
                    y=["alzheimerbroadcategory_bin"],
                )
            ),
            (BadUserInput, "Data model .* does not exist."),
        ),
        (
            "test_algorithm1",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model1:0.1",
                    datasets=["non_existing", "non_existing2"],
                    x=["lefthippocampus", "righthippocampus"],
                    y=["alzheimerbroadcategory_bin"],
                )
            ),
            (
                BadUserInput,
                "Datasets:.* could not be found for data_model:.*",
            ),
        ),
        (
            "test_algorithm1",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model1:0.1",
                    datasets=["test_dataset1"],
                    x=["test_cde1", "test_cde2"],
                    y=["test_cde3", "test_cde2"],
                )
            ),
            (BadUserInput, "Inputdata .* cannot have multiple values."),
        ),
        (
            "test_algorithm1",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model1:0.1",
                    datasets=["test_dataset1"],
                    x=["test_cde1", "test_cde2"],
                    y=["non_existing"],
                )
            ),
            (BadUserInput, "The CDE .* does not exist in data model .*"),
        ),
        (
            "test_algorithm1",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model1:0.1",
                    datasets=["test_dataset1"],
                    x=["test_cde1", "test_cde2"],
                    y=["test_cde1"],
                )
            ),
            (BadUserInput, "The CDE .* doesn't have one of the allowed types .*"),
        ),
        (
            "test_algorithm1",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model1:0.1",
                    datasets=["test_dataset1"],
                    x=["test_cde1", "test_cde2"],
                    y=["test_cde4"],
                )
            ),
            (BadUserInput, "The CDE .* should be categorical."),
        ),
        (
            "test_algorithm1",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model1:0.1",
                    datasets=["test_dataset1"],
                    x=["test_cde5", "test_cde2"],
                    y=["test_cde3"],
                ),
            ),
            (BadUserInput, "The CDE .* should NOT be categorical."),
        ),
        (
            "test_algorithm1",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model1:0.1",
                    datasets=["test_dataset1"],
                    x=["test_cde1", "test_cde2"],
                    y=["test_cde6"],
                ),
            ),
            (BadUserInput, "The CDE .* should have .* enumerations."),
        ),
        (
            "test_algorithm1",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model1:0.1",
                    datasets=["test_dataset1"],
                    x=["test_cde1", "test_cde2"],
                    y=["test_cde3"],
                ),
            ),
            (BadUserInput, "Algorithm parameters not provided."),
        ),
        (
            "test_algorithm1",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model1:0.1",
                    datasets=["test_dataset1"],
                    x=["test_cde1", "test_cde2"],
                    y=["test_cde3"],
                ),
                parameters={"wrong_parameter": ""},
            ),
            (BadUserInput, "Parameter .* should not be blank."),
        ),
        (
            "test_algorithm1",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model1:0.1",
                    datasets=["test_dataset1"],
                    x=["test_cde1", "test_cde2"],
                    y=["test_cde3"],
                ),
                parameters={"parameter1": 2},
            ),
            (BadUserInput, "Parameter .* should be a list."),
        ),
        (
            "test_algorithm1",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model1:0.1",
                    datasets=["test_dataset1"],
                    x=["test_cde1", "test_cde2"],
                    y=["test_cde3"],
                ),
                parameters={"parameter1": [1, 3], "parameter4": [1, 2.3]},
            ),
            (BadUserInput, "Parameter .* values should be of type .*"),
        ),
        (
            "test_algorithm1",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model1:0.1",
                    datasets=["test_dataset1"],
                    x=["test_cde1", "test_cde2"],
                    y=["test_cde3"],
                ),
                parameters={"parameter1": [1, 3], "parameter2": "wrong"},
            ),
            (BadUserInput, "Parameter .* values should be of type .*"),
        ),
        (
            "test_algorithm1",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model1:0.1",
                    datasets=["test_dataset1"],
                    x=["test_cde1", "test_cde2"],
                    y=["test_cde3"],
                ),
                parameters={"parameter1": [1, 3], "parameter3": 1},
            ),
            (BadUserInput, "Parameter .* values should be of type .*"),
        ),
        (
            "test_algorithm1",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model1:0.1",
                    datasets=["test_dataset1"],
                    x=["test_cde1", "test_cde2"],
                    y=["test_cde3"],
                ),
                parameters={"parameter1": [1, 4]},
            ),
            (BadUserInput, "Parameter .* values should be one of the following: .*"),
        ),
        (
            "test_algorithm1",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model1:0.1",
                    datasets=["test_dataset1"],
                    x=["test_cde1", "test_cde2"],
                    y=["test_cde3"],
                ),
                parameters={"parameter1": [1], "parameter2": 1},
            ),
            (BadUserInput, "Parameter .* values should be greater than .*"),
        ),
        (
            "test_algorithm1",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model1:0.1",
                    datasets=["test_dataset1"],
                    x=["test_cde1", "test_cde2"],
                    y=["test_cde3"],
                ),
                parameters={"parameter1": [1], "parameter2": 10},
            ),
            (BadUserInput, "Parameter .* values should be at most equal to .*"),
        ),
        (
            "algorithm_without_x",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model2:0.1",
                    datasets=["test_dataset2", "test_dataset3"],
                ),
            ),
            (BadUserInput, "Inputdata .* should be provided."),
        ),
    ]
    return parametrization_list


@pytest.mark.parametrize(
    "algorithm_name, request_dto, exception", get_parametrization_list_exception_cases()
)
def test_validate_algorithm_exceptions(
    algorithm_name, request_dto, exception, available_datasets_per_data_model
):
    exception_type, exception_message = exception
    with pytest.raises(exception_type, match=exception_message):
        validate_algorithm_request(
            algorithm_name=algorithm_name,
            algorithm_request_dto=request_dto,
            available_datasets_per_data_model=available_datasets_per_data_model,
        )
