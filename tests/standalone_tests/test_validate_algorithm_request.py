from unittest.mock import patch

import pytest

from mipengine.common_data_elements import CommonDataElement
from mipengine.common_data_elements import CommonDataElements
from mipengine.common_data_elements import MetadataEnumeration
from mipengine.common_data_elements import MetadataVariable
from mipengine.controller.algorithms_specifications import AlgorithmSpecifications
from mipengine.controller.algorithms_specifications import AlgorithmsSpecifications
from mipengine.controller.algorithms_specifications import ParameterSpecification
from mipengine.controller.algorithms_specifications import InputDataSpecification
from mipengine.controller.algorithms_specifications import InputDataSpecifications
from mipengine.controller.api.exceptions import BadRequest
from mipengine.controller.api.exceptions import BadUserInput
from mipengine.controller.api.validator import validate_algorithm_request

from mipengine.controller.api.algorithm_request_dto import (
    AlgorithmInputDataDTO,
    AlgorithmRequestDTO,
)


@pytest.fixture(scope="module", autouse=True)
def mock_cdes():
    common_data_elements = CommonDataElements()
    common_data_elements.pathologies = {
        "test_data_model1": {
            "test_cde1": CommonDataElement(
                MetadataVariable(
                    code="test_cde1",
                    label="test cde1",
                    sql_type="int",
                    isCategorical=False,
                    enumerations=None,
                    min=None,
                    max=None,
                )
            ),
            "test_cde2": CommonDataElement(
                MetadataVariable(
                    code="test_cde2",
                    label="test cde2",
                    sql_type="real",
                    isCategorical=False,
                    enumerations=None,
                    min=None,
                    max=None,
                )
            ),
            "test_cde3": CommonDataElement(
                MetadataVariable(
                    code="test_cde3",
                    label="test cde3",
                    sql_type="text",
                    isCategorical=True,
                    enumerations=[
                        MetadataEnumeration(code="male", label="male"),
                        MetadataEnumeration(code="female", label="female"),
                    ],
                    min=None,
                    max=None,
                )
            ),
            "test_cde4": CommonDataElement(
                MetadataVariable(
                    code="test_cde4",
                    label="test cde4",
                    sql_type="text",
                    isCategorical=False,
                    min=None,
                    max=None,
                )
            ),
            "test_cde5": CommonDataElement(
                MetadataVariable(
                    code="test_cde5",
                    label="test cde5",
                    sql_type="int",
                    isCategorical=True,
                    enumerations=[
                        MetadataEnumeration(code="1", label="1"),
                        MetadataEnumeration(code="2", label="2"),
                    ],
                    min=None,
                    max=None,
                )
            ),
            "test_cde6": CommonDataElement(
                MetadataVariable(
                    code="test_cde6",
                    label="test cde6",
                    sql_type="text",
                    isCategorical=True,
                    enumerations=[
                        MetadataEnumeration(code="male", label="male"),
                        MetadataEnumeration(code="female", label="female"),
                        MetadataEnumeration(code="Other", label="Other"),
                    ],
                    min=None,
                    max=None,
                )
            ),
        },
        "test_data_model2": {
            "test_cde1": CommonDataElement(
                MetadataVariable(
                    code="test_cde1",
                    label="test cde1",
                    sql_type="int",
                    isCategorical=False,
                    enumerations=None,
                    min=None,
                    max=None,
                )
            ),
            "test_cde2": CommonDataElement(
                MetadataVariable(
                    code="test_cde2",
                    label="test cde2",
                    sql_type="real",
                    isCategorical=False,
                    enumerations=None,
                    min=None,
                    max=None,
                )
            ),
            "test_cde3": CommonDataElement(
                MetadataVariable(
                    code="test_cde3",
                    label="test cde3",
                    sql_type="text",
                    isCategorical=True,
                    enumerations=[
                        MetadataEnumeration(code="male", label="male"),
                        MetadataEnumeration(code="female", label="female"),
                    ],
                    min=None,
                    max=None,
                )
            ),
        },
    }

    with patch(
        "mipengine.controller.api.validator.controller_common_data_elements",
        common_data_elements,
    ):
        yield


@pytest.fixture()
def available_datasets_per_data_model():
    d = {
        "test_data_model1": ["test_dataset1", "test_dataset2"],
        "test_data_model2": ["test_dataset2", "test_dataset3"],
    }
    return d


@pytest.fixture(scope="module", autouse=True)
def mock_algorithms_specs():
    algorithms_specifications = AlgorithmsSpecifications()
    algorithms_specifications.enabled_algorithms = {
        "test_algorithm1": AlgorithmSpecifications(
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
        "algorithm_without_y": AlgorithmSpecifications(
            name="algorithm_without_y",
            desc="algorithm_without_y",
            label="algorithm_without_y",
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
            ),
        ),
        "algorithm_without_x": AlgorithmSpecifications(
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
        "mipengine.controller.api.validator.algorithms_specifications",
        algorithms_specifications,
    ):
        yield


def get_parametrization_list_success_cases():
    parametrization_list = [
        (
            "test_algorithm1",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model1",
                    data_model_version="0.1",
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
                    data_model="test_data_model2",
                    data_model_version="0.1",
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
                    data_model="test_data_model2",
                    data_model_version="0.1",
                    datasets=["test_dataset2", "test_dataset3"],
                    y=["test_cde1"],
                ),
            ),
        ),
        (
            "algorithm_without_y",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model2",
                    data_model_version="0.1",
                    datasets=["test_dataset2", "test_dataset3"],
                    x=["test_cde1"],
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
                    data_model="test_data_model1",
                    data_model_version="0.1",
                    datasets=["demo_data"],
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
                    data_model="test_data_model1",
                    data_model_version="0.1",
                    datasets=["demo_data"],
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
                    data_model_version="0.1",
                    datasets=["demo_data"],
                    x=["lefthippocampus", "righthippocampus"],
                    y=["alzheimerbroadcategory_bin"],
                )
            ),
            (BadUserInput, "data_model .* does not exist."),
        ),
        (
            "test_algorithm1",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model1",
                    data_model_version="0.1",
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
                    data_model="test_data_model1",
                    data_model_version="0.1",
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
                    data_model="test_data_model1",
                    data_model_version="0.1",
                    datasets=["test_dataset1"],
                    x=["test_cde1", "test_cde2"],
                    y=["non_existing"],
                )
            ),
            (BadUserInput, "The CDE .* does not exist in data_model .*"),
        ),
        (
            "test_algorithm1",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="test_data_model1",
                    data_model_version="0.1",
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
                    data_model="test_data_model1",
                    data_model_version="0.1",
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
                    data_model="test_data_model1",
                    data_model_version="0.1",
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
                    data_model="test_data_model1",
                    data_model_version="0.1",
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
                    data_model="test_data_model1",
                    data_model_version="0.1",
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
                    data_model="test_data_model1",
                    data_model_version="0.1",
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
                    data_model="test_data_model1",
                    data_model_version="0.1",
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
                    data_model="test_data_model1",
                    data_model_version="0.1",
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
                    data_model="test_data_model1",
                    data_model_version="0.1",
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
                    data_model="test_data_model1",
                    data_model_version="0.1",
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
                    data_model="test_data_model1",
                    data_model_version="0.1",
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
                    data_model="test_data_model1",
                    data_model_version="0.1",
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
                    data_model="test_data_model1",
                    data_model_version="0.1",
                    datasets=["test_dataset1"],
                    x=["test_cde1", "test_cde2"],
                    y=["test_cde3"],
                ),
                parameters={"parameter1": [1], "parameter2": 10},
            ),
            (BadUserInput, "Parameter .* values should be less than .*"),
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
