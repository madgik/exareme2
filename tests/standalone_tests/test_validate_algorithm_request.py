from unittest.mock import patch

import pytest

from mipengine.controller.algorithm_specifications import AlgorithmSpecification
from mipengine.controller.algorithm_specifications import AlgorithmSpecifications
from mipengine.controller.algorithm_specifications import InputDataSpecification
from mipengine.controller.algorithm_specifications import InputDataSpecifications
from mipengine.controller.algorithm_specifications import ParameterEnumSpecification
from mipengine.controller.algorithm_specifications import ParameterSpecification
from mipengine.controller.api.algorithm_request_dto import AlgorithmInputDataDTO
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO
from mipengine.controller.api.algorithm_specifications_dtos import ParameterEnumType
from mipengine.controller.api.validator import BadRequest
from mipengine.controller.api.validator import validate_algorithm_request
from mipengine.controller.node_landscape_aggregator import DataModelRegistry
from mipengine.controller.node_landscape_aggregator import DataModelsCDES
from mipengine.controller.node_landscape_aggregator import NodeLandscapeAggregator
from mipengine.controller.node_landscape_aggregator import _NLARegistries
from mipengine.exceptions import BadUserInput
from mipengine.node_tasks_DTOs import CommonDataElement
from mipengine.node_tasks_DTOs import CommonDataElements


@pytest.fixture(scope="module", autouse=True)
def mock_cdes():
    nla = NodeLandscapeAggregator()
    data_models = {
        "data_model_with_all_cde_types:0.1": CommonDataElements(
            values={
                "int_cde": CommonDataElement(
                    code="int_cde",
                    label="int_cde",
                    sql_type="int",
                    is_categorical=False,
                ),
                "real_cde": CommonDataElement(
                    code="real_cde",
                    label="real_cde",
                    sql_type="real",
                    is_categorical=False,
                ),
                "text_cde_categ": CommonDataElement(
                    code="text_cde_categ",
                    label="text_cde_categ",
                    sql_type="text",
                    is_categorical=True,
                    enumerations={"male": "male", "female": "female"},
                ),
                "text_cde_non_categ": CommonDataElement(
                    code="text_cde_non_categ",
                    label="text_cde_non_categ",
                    sql_type="text",
                    is_categorical=False,
                ),
                "int_cde_categ": CommonDataElement(
                    code="int_cde_categ",
                    label="int_cde_categ",
                    sql_type="int",
                    is_categorical=True,
                    enumerations={
                        "1": "1",
                        "2": "2",
                    },
                ),
                "text_cde_3_enums": CommonDataElement(
                    code="text_cde_3_enums",
                    label="text_cde_3_enums",
                    sql_type="text",
                    is_categorical=True,
                    enumerations={"male": "male", "female": "female", "Other": "Other"},
                ),
            }
        ),
        "sample_data_model:0.1": CommonDataElements(
            values={
                "sample_cde": CommonDataElement(
                    code="sample_cde",
                    label="sample_cde",
                    sql_type="int",
                    is_categorical=False,
                ),
            }
        ),
    }
    _data_model_registry = DataModelRegistry(
        data_models_cdes=DataModelsCDES(data_models_cdes=data_models),
    )
    nla._registries = _NLARegistries(data_model_registry=_data_model_registry)

    with patch(
        "mipengine.controller.api.validator.node_landscape_aggregator",
        nla,
    ):
        yield


@pytest.fixture()
def available_datasets_per_data_model():
    d = {
        "data_model_with_all_cde_types:0.1": ["sample_dataset1", "sample_dataset2"],
        "sample_data_model:0.1": ["sample_dataset"],
    }
    return d


@pytest.fixture(scope="module", autouse=True)
def mock_algorithms_specs():
    algorithms_specifications = AlgorithmSpecifications()
    algorithms_specifications.enabled_algorithms = {
        "algorithm_with_y_int": AlgorithmSpecification(
            name="algorithm_with_y_int",
            desc="algorithm_with_y_int",
            label="algorithm_with_y_int",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="features",
                    desc="Features",
                    types=["real"],
                    stattypes=["numerical"],
                    notblank=True,
                    multiple=False,
                ),
            ),
        ),
        "algorithm_with_x_int_and_y_text": AlgorithmSpecification(
            name="algorithm_with_x_int_and_y_text",
            desc="algorithm_with_x_int_and_y_text",
            label="algorithm_with_x_int_and_y_text",
            enabled=True,
            inputdata=InputDataSpecifications(
                x=InputDataSpecification(
                    label="features",
                    desc="Features",
                    types=["real"],
                    stattypes=["numerical"],
                    notblank=True,
                    multiple=False,
                ),
                y=InputDataSpecification(
                    label="target",
                    desc="Target variable",
                    types=["text"],
                    stattypes=["nominal"],
                    notblank=True,
                    multiple=False,
                ),
            ),
        ),
        "algorithm_with_y_text_multiple_true": AlgorithmSpecification(
            name="algorithm_with_y_text_multiple_true",
            desc="algorithm_with_y_text_multiple_true",
            label="algorithm_with_y_text_multiple_true",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="target",
                    desc="Target variable",
                    types=["text"],
                    stattypes=["nominal", "numerical"],
                    notblank=True,
                    multiple=True,
                ),
            ),
        ),
        "algorithm_with_y_text_categ": AlgorithmSpecification(
            name="algorithm_with_y_text_categ",
            desc="algorithm_with_y_text_categ",
            label="algorithm_with_y_text_categ",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="target",
                    desc="Target variable",
                    types=["text"],
                    stattypes=["nominal"],
                    notblank=True,
                    multiple=True,
                ),
            ),
        ),
        "algorithm_with_y_text_non_categ": AlgorithmSpecification(
            name="algorithm_with_y_text_non_categ",
            desc="algorithm_with_y_text_non_categ",
            label="algorithm_with_y_text_non_categ",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="target",
                    desc="Target variable",
                    types=["text"],
                    stattypes=["numerical"],
                    notblank=True,
                    multiple=True,
                ),
            ),
        ),
        "algorithm_with_variable_enumslen": AlgorithmSpecification(
            name="algorithm_with_variable_enumslen",
            desc="algorithm_with_variable_enumslen",
            label="algorithm_with_variable_enumslen",
            enabled=True,
            inputdata=InputDataSpecifications(
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
        ),
        "algorithm_with_y_and_x_optional": AlgorithmSpecification(
            name="algorithm_with_y_and_x_optional",
            desc="algorithm_with_y_and_x_optional",
            label="algorithm_with_y_and_x_optional",
            enabled=True,
            inputdata=InputDataSpecifications(
                x=InputDataSpecification(
                    label="features",
                    desc="Features",
                    types=["real"],
                    stattypes=["numerical"],
                    notblank=False,
                    multiple=False,
                ),
                y=InputDataSpecification(
                    label="target",
                    desc="Target variable",
                    types=["text"],
                    stattypes=["nominal"],
                    notblank=True,
                    multiple=False,
                ),
            ),
        ),
        "algorithm_with_required_param": AlgorithmSpecification(
            name="algorithm_with_required_param",
            desc="algorithm_with_required_param",
            label="algorithm_with_required_param",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="features",
                    desc="Features",
                    types=["real"],
                    stattypes=["numerical"],
                    notblank=True,
                    multiple=False,
                ),
            ),
            parameters={
                "required_param": ParameterSpecification(
                    label="required_param",
                    desc="required_param",
                    types=["real"],
                    notblank=True,
                    multiple=False,
                ),
            },
        ),
        "algorithm_with_many_params": AlgorithmSpecification(
            name="algorithm_with_x_and_y",
            desc="algorithm_with_x_and_y",
            label="algorithm_with_x_and_y",
            enabled=True,
            inputdata=InputDataSpecifications(
                y=InputDataSpecification(
                    label="target",
                    desc="Target variable",
                    types=["text"],
                    stattypes=["nominal"],
                    notblank=True,
                    multiple=False,
                ),
            ),
            parameters={
                "int_parameter_with_min_max": ParameterSpecification(
                    label="parameter_with_min_max",
                    desc="parameter_with_min_max",
                    types=["int"],
                    notblank=False,
                    multiple=False,
                    min=2,
                    max=5,
                ),
                "text_parameter": ParameterSpecification(
                    label="text_parameter",
                    desc="text_parameter",
                    types=["text"],
                    notblank=False,
                    multiple=False,
                ),
                "parameter_multiple_true": ParameterSpecification(
                    label="parameter_multiple_true",
                    desc="parameter_multiple_true",
                    types=["int"],
                    notblank=False,
                    multiple=True,
                ),
                "param_with_enum_type_list": ParameterSpecification(
                    label="param_with_enum_type_list",
                    desc="param_with_enum_type_list",
                    types=["text"],
                    notblank=False,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.LIST,
                        source=["a", "b", "c"],
                    ),
                ),
                "param_with_enum_type_list_multiple_true": ParameterSpecification(
                    label="param_with_enum_type_list_multiple_true",
                    desc="param_with_enum_type_list_multiple_true",
                    types=["text"],
                    notblank=False,
                    multiple=True,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.LIST,
                        source=["a", "b", "c"],
                    ),
                ),
                "param_with_enum_type_inputdata_cde_enums": ParameterSpecification(
                    label="param_with_enum_type_inputdata_cde_enums",
                    desc="param_with_enum_type_inputdata_cde_enums",
                    types=["text"],
                    notblank=False,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.INPUTDATA_CDE_ENUMS,
                        source="y",
                    ),
                ),
                "param_with_enum_type_cde_enums": ParameterSpecification(
                    label="param_with_enum_type_cde_enums",
                    desc="param_with_enum_type_cde_enums",
                    types=["text"],
                    notblank=False,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.CDE_ENUMS,
                        source="text_cde_categ",
                    ),
                ),
                "param_with_enum_type_cde_enums_wrong_CDE": ParameterSpecification(
                    label="cde_enums_param_wrong_CDE",
                    desc="parameter that uses enums with type CDE_enums",
                    types=["text"],
                    notblank=False,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.CDE_ENUMS,
                        source="non_existing_CDE",
                    ),
                ),
                "param_with_enum_type_inputdata_CDEs": ParameterSpecification(
                    label="cde_enums_param_inputdata_CDEs",
                    desc="parameter that uses enums with type inputdata_CDEs",
                    types=["text"],
                    notblank=False,
                    multiple=False,
                    enums=ParameterEnumSpecification(
                        type=ParameterEnumType.INPUTDATA_CDES,
                        source=["x", "y"],
                    ),
                ),
            },
        ),
    }

    with patch(
        "mipengine.controller.api.validator.algorithm_specifications",
        algorithms_specifications,
    ):
        yield


def get_parametrization_list_success_cases():
    parametrization_list = [
        pytest.param(
            "algorithm_with_y_int",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1", "sample_dataset2"],
                    y=["int_cde"],
                ),
            ),
            id="multiple datasets",
        ),
        pytest.param(
            "algorithm_with_x_int_and_y_text",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    x=["int_cde"],
                    y=["text_cde_categ"],
                ),
            ),
            id="proper variable types",
        ),
        pytest.param(
            "algorithm_with_y_text_multiple_true",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ", "text_cde_non_categ"],
                ),
            ),
            id="variables multiple true",
        ),
        pytest.param(
            "algorithm_with_variable_enumslen",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
            ),
            id="variable enumslen",
        ),
        pytest.param(
            "algorithm_with_y_and_x_optional",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
            ),
            id="variable notblank false",
        ),
        pytest.param(
            "algorithm_with_y_and_x_optional",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
            ),
            id="variable notblank false",
        ),
        pytest.param(
            "algorithm_with_required_param",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["int_cde"],
                ),
                parameters={"required_param": 1},
            ),
            id="parameter required",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"int_parameter_with_min_max": 2},
            ),
            id="parameter min max",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"parameter_multiple_true": [1, 2, 3]},
            ),
            id="parameter multiple true",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"param_with_enum_type_list": "a"},
            ),
            id="parameter enums type list",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"param_with_enum_type_list_multiple_true": ["a", "c"]},
            ),
            id="parameter enums type list and multiple true",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"param_with_enum_type_inputdata_cde_enums": "male"},
            ),
            id="parameter enums type inputdata_CDE_enums",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"param_with_enum_type_cde_enums": "female"},
            ),
            id="parameter enums type CDE_enums",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"param_with_enum_type_inputdata_CDEs": "text_cde_categ"},
            ),
            id="parameter enums type inputdata_CDEs",
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
        pytest.param(
            "non_existing_algorithm",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                )
            ),
            (BadRequest, "Algorithm .* does not exist."),
            id="Algorithm does not exist.",
        ),
        pytest.param(
            "algorithm_with_y_int",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["non_existing_dataset"],
                    y=["int_cde"],
                )
            ),
            (
                BadUserInput,
                "Datasets:.* could not be found for data_model:.*",
            ),
            id="Dataset does not exist.",
        ),
        pytest.param(
            "algorithm_with_y_int",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset"],
                    y=["int_cde"],
                )
            ),
            (
                BadUserInput,
                "Datasets:.* could not be found for data_model:.*",
            ),
            id="Dataset does not exist in specified data model.",
        ),
        pytest.param(
            "algorithm_with_y_int",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="non_existing_data_model:0.1",
                    datasets=["sample_dataset"],
                    y=["int_cde"],
                )
            ),
            (BadUserInput, "Data model .* does not exist."),
            id="Data model does not exist.",
        ),
        pytest.param(
            "algorithm_with_y_int",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                ),
            ),
            (BadUserInput, "Inputdata .* should be provided."),
            id="Inputdata not provided.",
        ),
        pytest.param(
            "algorithm_with_y_int",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ", "real_cde"],
                )
            ),
            (BadUserInput, "Inputdata .* cannot have multiple values."),
            id="Inputdata variable with multiple false given list.",
        ),
        pytest.param(
            "algorithm_with_y_int",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["non_existing"],
                )
            ),
            (BadUserInput, "The CDE .* does not exist in the data model provided."),
            id="CDE does not exist.",
        ),
        pytest.param(
            "algorithm_with_y_int",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                )
            ),
            (BadUserInput, "The CDE .* doesn't have one of the allowed types .*"),
            id="Inputdata variable wrong CDE type.",
        ),
        pytest.param(
            "algorithm_with_y_text_categ",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_non_categ"],
                )
            ),
            (BadUserInput, "The CDE .* should be categorical."),
            id="Inputdata variable requires categorical CDE given non categorical.",
        ),
        pytest.param(
            "algorithm_with_y_text_non_categ",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
            ),
            (BadUserInput, "The CDE .* should NOT be categorical."),
            id="Inputdata variable requires non categorical CDE given categorical.",
        ),
        pytest.param(
            "algorithm_with_variable_enumslen",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_3_enums"],
                ),
            ),
            (BadUserInput, "The CDE .* should have .* enumerations."),
            id="Inputdata variable requires 2 enumerations, CDE has 3 enums.",
        ),
        pytest.param(
            "algorithm_with_required_param",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["int_cde"],
                ),
            ),
            (BadUserInput, "Algorithm parameters not provided."),
            id="Required algorithm parameter not provided.",
        ),
        pytest.param(
            "algorithm_with_required_param",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["int_cde"],
                ),
                parameters={"wrong_parameter": ""},
            ),
            (BadUserInput, "Parameter .* should not be blank."),
            id="Required parameter not provided.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"parameter_multiple_true": 2},
            ),
            (BadUserInput, "Parameter .* should be a list."),
            id="Parameter with multiple=true given singular value.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"text_parameter": 1},
            ),
            (BadUserInput, "Parameter .* values should be of types.*"),
            id="Parameter of type text given int value.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"int_parameter_with_min_max": "text_value"},
            ),
            (BadUserInput, "Parameter .* values should be of types.*"),
            id="Parameter of type int given text value.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"int_parameter_with_min_max": [1, 2, 3]},
            ),
            (BadUserInput, "Parameter .* values should be of types.*"),
            id="Parameter of type int with multiple=false given list of ints.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"param_with_enum_type_list": "non_existing_enum"},
            ),
            (BadUserInput, "Parameter .* values should be one of the following: .*"),
            id="Parameter with enumerations of type list given non existing enum.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"int_parameter_with_min_max": 1},
            ),
            (BadUserInput, "Parameter .* values should be greater than .*"),
            id="Parameter with min max given a lesser value.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={"int_parameter_with_min_max": 10},
            ),
            (BadUserInput, "Parameter .* values should be at most equal to .*"),
            id="Parameter with min max given a greater value.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={
                    "param_with_enum_type_inputdata_cde_enums": "non_existing_enum",
                },
            ),
            (
                BadUserInput,
                "Parameter's .* enums, that are taken from the CDE .* given in inputdata .* variable, should be one of the following: .*",
            ),
            id="Parameter with enumerations of type inputdata_CDE_enums given non existing enum.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={
                    "param_with_enum_type_cde_enums_wrong_CDE": "male",
                },
            ),
            (
                ValueError,
                "Parameter's .* enums source .* does not exist in the data model provided.",
            ),
            id="Parameter with enumerations of type CDE_enums has, in the algorithm specification, a CDE that doesn't exist.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={
                    "param_with_enum_type_cde_enums": "non_existing_enum",
                },
            ),
            (
                BadUserInput,
                "Parameter's .* enums, that are taken from the CDE .*, should be one of the following: .*",
            ),
            id="Parameter with enumerations of type CDE_enums given non existing enum.",
        ),
        pytest.param(
            "algorithm_with_many_params",
            AlgorithmRequestDTO(
                inputdata=AlgorithmInputDataDTO(
                    data_model="data_model_with_all_cde_types:0.1",
                    datasets=["sample_dataset1"],
                    y=["text_cde_categ"],
                ),
                parameters={
                    "param_with_enum_type_inputdata_CDEs": "text_cde_non_categ",
                },
            ),
            (
                BadUserInput,
                "Parameter's .* enums, that are taken from inputdata .* CDEs, should be one of the following: .*",
            ),
            id="Parameter with enumerations of type inputdata_CDEs given non existing enum.",
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
