import numbers
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from mipengine.controller import config as ctrl_config
from mipengine.controller.algorithm_specifications import AlgorithmSpecification
from mipengine.controller.algorithm_specifications import InputDataSpecification
from mipengine.controller.algorithm_specifications import InputDataSpecifications
from mipengine.controller.algorithm_specifications import InputDataStatType
from mipengine.controller.algorithm_specifications import InputDataType
from mipengine.controller.algorithm_specifications import ParameterSpecification
from mipengine.controller.algorithm_specifications import algorithm_specifications
from mipengine.controller.api.algorithm_request_dto import USE_SMPC_FLAG
from mipengine.controller.api.algorithm_request_dto import AlgorithmInputDataDTO
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO
from mipengine.controller.api.algorithm_specifications_dtos import ParameterEnumType
from mipengine.controller.node_landscape_aggregator import NodeLandscapeAggregator
from mipengine.exceptions import BadUserInput
from mipengine.filters import validate_filter
from mipengine.node_tasks_DTOs import CommonDataElement
from mipengine.smpc_cluster_comm_helpers import validate_smpc_usage

node_landscape_aggregator = NodeLandscapeAggregator()


class BadRequest(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


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
    if algorithm_name not in algorithm_specifications.enabled_algorithms.keys():
        raise BadRequest(f"Algorithm '{algorithm_name}' does not exist.")
    return algorithm_specifications.enabled_algorithms[algorithm_name]


def _validate_algorithm_request_body(
    algorithm_request_dto: AlgorithmRequestDTO,
    algorithm_specs: AlgorithmSpecification,
    available_datasets_per_data_model: Dict[str, List[str]],
):
    _validate_data_model(
        requested_data_model=algorithm_request_dto.inputdata.data_model,
        available_datasets_per_data_model=available_datasets_per_data_model,
    )

    data_model_cdes = node_landscape_aggregator.get_cdes(
        algorithm_request_dto.inputdata.data_model
    )

    _validate_inputdata(
        inputdata=algorithm_request_dto.inputdata,
        inputdata_specs=algorithm_specs.inputdata,
        available_datasets_per_data_model=available_datasets_per_data_model,
        data_model_cdes=data_model_cdes,
    )

    _validate_parameters(
        algorithm_request_dto.parameters,
        algorithm_specs.parameters,
        algorithm_request_dto.inputdata,
        data_model_cdes=data_model_cdes,
    )

    _validate_flags(algorithm_request_dto.flags)


def _validate_data_model(requested_data_model: str, available_datasets_per_data_model):
    if requested_data_model not in available_datasets_per_data_model.keys():
        raise BadUserInput(f"Data model '{requested_data_model}' does not exist.")


def _validate_inputdata(
    inputdata: AlgorithmInputDataDTO,
    inputdata_specs: InputDataSpecifications,
    available_datasets_per_data_model: Dict[str, List[str]],
    data_model_cdes: Dict[str, CommonDataElement],
):
    _validate_inputdata_dataset(
        requested_data_model=inputdata.data_model,
        requested_datasets=inputdata.datasets,
        available_datasets_per_data_model=available_datasets_per_data_model,
    )
    _validate_inputdata_filter(inputdata.data_model, inputdata.filters, data_model_cdes)
    _validate_algorithm_inputdatas(inputdata, inputdata_specs, data_model_cdes)


def _validate_inputdata_dataset(
    requested_data_model: str,
    requested_datasets: List[str],
    available_datasets_per_data_model: Dict[str, List[str]],
):
    """
    Validates that the dataset values exist and that the datasets belong in the data_model.
    """
    non_existing_datasets = [
        dataset
        for dataset in requested_datasets
        if dataset not in available_datasets_per_data_model[requested_data_model]
    ]
    if non_existing_datasets:
        raise BadUserInput(
            f"Datasets:'{non_existing_datasets}' could not be found for data_model:{requested_data_model}"
        )


def _validate_inputdata_filter(data_model, filter, data_model_cdes):
    """
    Validates that the filter provided have the correct format
    following: https://querybuilder.js.org/
    """
    validate_filter(data_model, filter, data_model_cdes)


def _validate_algorithm_inputdatas(
    inputdata: AlgorithmInputDataDTO,
    inputdata_specs: InputDataSpecifications,
    data_model_cdes: Dict[str, CommonDataElement],
):

    if inputdata_specs.x:
        _validate_algorithm_inputdata(inputdata.x, inputdata_specs.x, data_model_cdes)
    if inputdata_specs.y:
        _validate_algorithm_inputdata(inputdata.y, inputdata_specs.y, data_model_cdes)


def _validate_algorithm_inputdata(
    inputdata_values: Optional[List[str]],
    inputdata_spec: InputDataSpecification,
    data_model_cdes: Dict[str, CommonDataElement],
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
        _validate_inputdata_value(inputdata_value, inputdata_spec, data_model_cdes)


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
    inputdata_value: str,
    inputdata_specs: InputDataSpecification,
    data_model_cdes: Dict[str, CommonDataElement],
):
    inputdata_value_metadata = _get_cde_metadata(inputdata_value, data_model_cdes)
    _validate_inputdata_types(
        inputdata_value, inputdata_specs, inputdata_value_metadata
    )
    _validate_inputdata_stattypes(
        inputdata_value, inputdata_specs, inputdata_value_metadata
    )
    _validate_inputdata_enumerations(
        inputdata_value, inputdata_specs, inputdata_value_metadata
    )


def _get_cde_metadata(cde, data_model_cdes):
    if cde not in data_model_cdes.keys():
        raise BadUserInput(
            f"The CDE '{cde}' does not exist in the data model provided."
        )
    return data_model_cdes[cde]


def _validate_inputdata_types(
    inputdata_value: str,
    inputdata_specs: InputDataSpecification,
    inputdata_value_metadata: CommonDataElement,
):
    dtype = InputDataType(inputdata_value_metadata.sql_type)
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
    inputdata: AlgorithmInputDataDTO,
    data_model_cdes: Dict[str, CommonDataElement],
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
                parameter_values=parameter_values,
                parameter_spec=parameter_spec,
                inputdata=inputdata,
                data_model_cdes=data_model_cdes,
            )


def _validate_parameter_values(
    parameter_values: Any,
    parameter_spec: ParameterSpecification,
    inputdata: AlgorithmInputDataDTO,
    data_model_cdes: Dict[str, CommonDataElement],
):
    if parameter_spec.multiple and not isinstance(parameter_values, list):
        raise BadUserInput(f"Parameter '{parameter_spec.label}' should be a list.")

    if not parameter_spec.multiple:
        parameter_values = [parameter_values]
    for parameter_value in parameter_values:
        _validate_parameter_type(parameter_value, parameter_spec)

        _validate_param_enums(
            parameter_value, parameter_spec, inputdata, data_model_cdes
        )

        _validate_parameter_inside_min_max(parameter_value, parameter_spec)


def _validate_parameter_type(
    parameter_value: Any,
    parameter_spec: ParameterSpecification,
):
    mip_types_to_python_types = {
        "text": str,
        "int": int,
        "real": numbers.Real,
        "boolean": bool,
    }

    for param_type in parameter_spec.types:
        if isinstance(parameter_value, mip_types_to_python_types[param_type.value]):
            return
    else:
        raise BadUserInput(
            f"Parameter '{parameter_spec.label}' values should be of types: {[type.value for type in parameter_spec.types]}."
        )


def _validate_param_enums(
    parameter_value: Any,
    parameter_spec: ParameterSpecification,
    inputdata: AlgorithmInputDataDTO,
    data_model_cdes: Dict[str, CommonDataElement],
):
    if parameter_spec.enums is None:
        return

    if parameter_spec.enums.type == ParameterEnumType.LIST:
        _validate_param_enums_of_type_list(parameter_value, parameter_spec)
    elif parameter_spec.enums.type == ParameterEnumType.INPUTDATA_CDE_ENUMS:
        _validate_param_enums_of_type_inputdata_CDE_enums(
            parameter_value, parameter_spec, inputdata, data_model_cdes
        )
    elif parameter_spec.enums.type == ParameterEnumType.CDE_ENUMS:
        _validate_param_enums_of_type_CDE_enums(
            parameter_value, parameter_spec, data_model_cdes
        )
    elif parameter_spec.enums.type == ParameterEnumType.INPUTDATA_CDES:
        _validate_param_enums_of_type_inputdata_CDEs(
            parameter_value, parameter_spec, inputdata
        )
    else:
        raise NotImplementedError(
            f"Parameter enum type not supported: '{parameter_spec.enums.type}'."
        )


def _validate_param_enums_of_type_inputdata_CDEs(
    parameter_value: Any,
    parameter_spec: ParameterSpecification,
    inputdata: AlgorithmInputDataDTO,
):
    inputdata_CDEs_enums = []
    inputdata_CDEs_enums.extend(inputdata.y) if inputdata.y else None
    inputdata_CDEs_enums.extend(inputdata.x) if inputdata.x else None
    if parameter_value not in inputdata_CDEs_enums:
        raise BadUserInput(
            f"Parameter's '{parameter_spec.label}' enums, that are taken from inputdata {parameter_spec.enums.source} CDEs, "
            f"should be one of the following: '{inputdata_CDEs_enums}'.",
        )


def _validate_param_enums_of_type_CDE_enums(
    parameter_value: Any,
    parameter_spec: ParameterSpecification,
    data_model_cdes: Dict[str, CommonDataElement],
):
    if parameter_spec.enums.source not in data_model_cdes.keys():
        raise ValueError(
            f"Parameter's '{parameter_spec.label}' enums source '{parameter_spec.enums.source}' does "
            f"not exist in the data model provided."
        )
    CDE_enums = list(data_model_cdes[parameter_spec.enums.source].enumerations.keys())
    if parameter_value not in CDE_enums:
        raise BadUserInput(
            f"Parameter's '{parameter_spec.label}' enums, that are taken from the CDE '{parameter_spec.enums.source}', "
            f"should be one of the following: '{list(CDE_enums)}'."
        )


def _validate_param_enums_of_type_inputdata_CDE_enums(
    parameter_value: Any,
    parameter_spec: ParameterSpecification,
    inputdata: AlgorithmInputDataDTO,
    data_model_cdes: Dict[str, CommonDataElement],
):
    if parameter_spec.enums.source == "x":
        inputdata_vars = inputdata.x
    elif parameter_spec.enums.source == "y":
        inputdata_vars = inputdata.y
    else:
        raise NotImplementedError(f"Source should be either 'x' or 'y'.")
    inputdata_var = inputdata_vars[0]  # multiple=true is not allowed
    inputdata_CDE_enums = data_model_cdes[inputdata_var].enumerations.keys()
    if parameter_value not in inputdata_CDE_enums:
        raise BadUserInput(
            f"Parameter's '{parameter_spec.label}' enums, that are taken from the CDE '{inputdata_var}' "
            f"given in inputdata '{parameter_spec.enums.source}' variable, "
            f"should be one of the following: '{list(inputdata_CDE_enums)}'."
        )


def _validate_param_enums_of_type_list(
    parameter_value: Any,
    parameter_spec: ParameterSpecification,
):
    if parameter_value not in parameter_spec.enums.source:
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
            f"should be at most equal to {parameter_spec.max} ."
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
