import numbers
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from mipengine.common_data_elements import CommonDataElement
from mipengine.common_data_elements import CommonDataElements
from mipengine.controller import config as ctrl_config
from mipengine.controller.algorithms_specifications import AlgorithmSpecifications
from mipengine.controller.algorithms_specifications import InputDataSpecification
from mipengine.controller.algorithms_specifications import InputDataSpecifications
from mipengine.controller.algorithms_specifications import InputDataStatType
from mipengine.controller.algorithms_specifications import InputDataType
from mipengine.controller.algorithms_specifications import ParameterSpecification
from mipengine.controller.algorithms_specifications import algorithms_specifications
from mipengine.controller.api.algorithm_request_dto import USE_SMPC_FLAG
from mipengine.controller.api.algorithm_request_dto import AlgorithmInputDataDTO
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO
from mipengine.controller.api.exceptions import BadRequest
from mipengine.controller.api.exceptions import BadUserInput
from mipengine.controller.controller_common_data_elements import get_cdes
from mipengine.filters import validate_filter
from mipengine.smpc_cluster_comm_helpers import validate_smpc_usage

# TODO This validator will be refactored heavily with https://team-1617704806227.atlassian.net/browse/MIP-90


def validate_algorithm_request(
    algorithm_name: str,
    algorithm_request_dto: AlgorithmRequestDTO,
    available_datasets_per_data_model: Dict[str, List[str]],
):
    algorithm_specs = _get_algorithm_specs(algorithm_name)
    _validate_algorithm_request_body(
        algorithm_request_dto=algorithm_request_dto,
        algorithm_specs=algorithm_specs,
        available_datasets_per_data_model=available_datasets_per_data_model,
    )


def _get_algorithm_specs(algorithm_name):
    if algorithm_name not in algorithms_specifications.enabled_algorithms.keys():
        raise BadRequest(f"Algorithm '{algorithm_name}' does not exist.")
    return algorithms_specifications.enabled_algorithms[algorithm_name]


def _validate_algorithm_request_body(
    algorithm_request_dto: AlgorithmRequestDTO,
    algorithm_specs: AlgorithmSpecifications,
    available_datasets_per_data_model: Dict[str, List[str]],
):
    _validate_inputdata(
        inputdata=algorithm_request_dto.inputdata,
        inputdata_specs=algorithm_specs.inputdata,
        available_datasets_per_data_model=available_datasets_per_data_model,
    )

    _validate_parameters(
        algorithm_request_dto.parameters,
        algorithm_specs.parameters,
    )

    _validate_flags(algorithm_request_dto.flags)


def _validate_inputdata(
    inputdata: AlgorithmInputDataDTO,
    inputdata_specs: InputDataSpecifications,
    available_datasets_per_data_model: Dict[str, List[str]],
):
    _validate_inputdata_data_model_and_dataset(
        requested_data_model=inputdata.data_model,
        requested_datasets=inputdata.datasets,
        available_datasets_per_data_model=available_datasets_per_data_model,
    )

    _validate_inputdata_filter(inputdata.data_model, inputdata.filters)
    _validate_algorithm_inputdatas(inputdata, inputdata_specs)


def _validate_inputdata_data_model_and_dataset(
    requested_data_model: str,
    requested_datasets: List[str],
    available_datasets_per_data_model: Dict[str, List[str]],
):
    """
    Validates that the data_model, dataset values exist and
    that the datasets belong in the data_model.
    """

    if requested_data_model not in available_datasets_per_data_model.keys():
        raise BadUserInput(f"data_model '{requested_data_model}' does not exist.")

    non_existing_datasets = [
        dataset
        for dataset in requested_datasets
        if dataset not in available_datasets_per_data_model[requested_data_model]
    ]
    if non_existing_datasets:
        raise BadUserInput(
            f"Datasets:'{non_existing_datasets}' could not be found for data_model:{requested_data_model}"
        )


def _validate_inputdata_filter(data_model, filter):
    """
    Validates that the filter provided have the correct format
    following: https://querybuilder.js.org/
    """
    common_data_elements = CommonDataElements(ctrl_config.cdes_metadata_path)
    validate_filter(common_data_elements, data_model, filter)


# TODO This will be removed with the dynamic inputdata logic.
def _validate_algorithm_inputdatas(
    inputdata: AlgorithmInputDataDTO, inputdata_specs: InputDataSpecifications
):

    _validate_algorithm_inputdata(inputdata.x, inputdata_specs.x, inputdata.data_model)
    _validate_algorithm_inputdata(inputdata.y, inputdata_specs.y, inputdata.data_model)


def _validate_algorithm_inputdata(
    inputdata_values: Optional[List[str]],
    inputdata_spec: InputDataSpecification,
    data_model: str,
):
    if not inputdata_values and not inputdata_spec:
        return

    if not inputdata_values:
        if inputdata_spec.notblank:
            raise BadUserInput(
                f"Inputdata '{inputdata_spec.label}' should be provided."
            )
        else:
            return

    _validate_inputdata_values_quantity(inputdata_values, inputdata_spec)

    for inputdata_value in inputdata_values:
        _validate_inputdata_value(inputdata_value, inputdata_spec, data_model)


def _validate_inputdata_values_quantity(
    inputdata_value: Any, inputdata_spec: InputDataSpecification
):
    if not isinstance(inputdata_value, list):
        raise BadRequest(f"Inputdata '{inputdata_spec.label}' should be a list.")

    if not inputdata_spec.multiple and len(inputdata_value) > 1:
        raise BadUserInput(
            f"Inputdata '{inputdata_spec.label}' cannot have multiple values."
        )


def _validate_inputdata_value(
    inputdata_value: str, inputdata_specs: InputDataSpecification, data_model: str
):
    inputdata_value_metadata = _get_cde_metadata(inputdata_value, data_model)
    _validate_inputdata_types(
        inputdata_value, inputdata_specs, inputdata_value_metadata
    )
    _validate_inputdata_stattypes(
        inputdata_value, inputdata_specs, inputdata_value_metadata
    )
    _validate_inputdata_enumerations(
        inputdata_value, inputdata_specs, inputdata_value_metadata
    )


def _get_cde_metadata(cde, data_model):
    data_model_cdes: Dict[str, CommonDataElement] = get_cdes()[data_model]
    if cde not in data_model_cdes.keys():
        raise BadUserInput(
            f"The CDE '{cde}' does not exist in data_model '{data_model}'."
        )
    return data_model_cdes[cde]


def _validate_inputdata_types(
    inputdata_value: str,
    inputdata_specs: InputDataSpecification,
    inputdata_value_metadata: CommonDataElement,
):
    dtype = inputdata_value_metadata.sql_type
    dtypes = inputdata_specs.types
    if dtype in dtypes:
        return
    if InputDataType.REAL in dtypes and dtype in (
        InputDataType.INT,
        InputDataType.REAL,
    ):
        return
    raise BadUserInput(
        f"The CDE '{inputdata_value}', of inputdata '{inputdata_specs.label}', "
        f"doesn't have one of the allowed types "
        f"'{inputdata_specs.types}'."
    )


def _validate_inputdata_stattypes(
    inputdata_value: str,
    inputdata_specs: InputDataSpecification,
    inputdata_value_metadata: CommonDataElement,
):
    can_be_numerical = InputDataStatType.NUMERICAL in inputdata_specs.stattypes
    can_be_nominal = InputDataStatType.NOMINAL in inputdata_specs.stattypes
    if not inputdata_value_metadata.is_categorical and not can_be_numerical:
        raise BadUserInput(
            f"The CDE '{inputdata_value}', of inputdata '{inputdata_specs.label}', "
            f"should be categorical."
        )
    if inputdata_value_metadata.is_categorical and not can_be_nominal:
        raise BadUserInput(
            f"The CDE '{inputdata_value}', of inputdata '{inputdata_specs.label}', "
            f"should NOT be categorical."
        )


def _validate_inputdata_enumerations(
    inputdata_value: str,
    inputdata_specs: InputDataSpecification,
    inputdata_value_metadata: CommonDataElement,
):
    if inputdata_specs.enumslen is not None and inputdata_specs.enumslen != len(
        inputdata_value_metadata.enumerations
    ):
        raise BadUserInput(
            f"The CDE '{inputdata_value}', of inputdata '{inputdata_specs.label}', "
            f"should have {inputdata_specs.enumslen} enumerations."
        )


def _validate_parameters(
    parameters: Optional[Dict[str, Any]],
    parameters_specs: Optional[Dict[str, ParameterSpecification]],
):
    """
    If the algorithm has parameters,
    it validates that they follow the algorithm specs.
    """
    if parameters_specs is None:
        return

    for parameter_name, parameter_spec in parameters_specs.items():
        if parameter_spec.notblank:
            if not parameters:
                raise BadUserInput(f"Algorithm parameters not provided.")
            if parameter_name not in parameters.keys():
                raise BadUserInput(f"Parameter '{parameter_name}' should not be blank.")

        parameter_values = parameters.get(parameter_name)
        if parameter_values:
            _validate_parameter_values(
                parameter_values,
                parameter_spec,
            )


def _validate_parameter_values(
    parameter_values: Any,
    parameter_spec: ParameterSpecification,
):
    if parameter_spec.multiple and not isinstance(parameter_values, list):
        raise BadUserInput(f"Parameter '{parameter_spec.label}' should be a list.")

    if not parameter_spec.multiple:
        parameter_values = [parameter_values]
    for parameter_value in parameter_values:
        _validate_parameter_type(parameter_value, parameter_spec)

        _validate_parameter_enumerations(parameter_value, parameter_spec)

        _validate_parameter_inside_min_max(parameter_value, parameter_spec)


def _validate_parameter_type(
    parameter_value: Any,
    parameter_spec: ParameterSpecification,
):
    mip_types_to_python_types = {
        "text": str,
        "int": int,
        "real": numbers.Real,
    }
    if not isinstance(parameter_value, mip_types_to_python_types[parameter_spec.type]):
        raise BadUserInput(
            f"Parameter '{parameter_spec.label}' values should be of type '{parameter_spec.type}'."
        )


def _validate_parameter_enumerations(
    parameter_value: Any,
    parameter_spec: ParameterSpecification,
):
    if parameter_spec.enums is None:
        return

    if parameter_value not in parameter_spec.enums:
        raise BadUserInput(
            f"Parameter '{parameter_spec.label}' values "
            f"should be one of the following: '{str(parameter_spec.enums)}'."
        )


def _validate_parameter_inside_min_max(
    parameter_value: Any,
    parameter_spec: ParameterSpecification,
):
    if parameter_spec.min is None and parameter_spec.max is None:
        return

    if parameter_spec.min is not None and parameter_value < parameter_spec.min:
        raise BadUserInput(
            f"Parameter '{parameter_spec.label}' values "
            f"should be greater than {parameter_spec.min} ."
        )

    if parameter_spec.max is not None and parameter_value > parameter_spec.max:
        raise BadUserInput(
            f"Parameter '{parameter_spec.label}' values "
            f"should be less than {parameter_spec.max} ."
        )


def _validate_flags(
    flags: Dict[str, Any],
):
    if not flags:
        return

    if USE_SMPC_FLAG in flags.keys():
        validate_smpc_usage(
            flags[USE_SMPC_FLAG], ctrl_config.smpc.enabled, ctrl_config.smpc.optional
        )
