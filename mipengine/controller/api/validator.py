import numbers
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from mipengine.common_data_elements import CommonDataElement, CommonDataElements
from mipengine.controller.controller_common_data_elements import (
    controller_common_data_elements,
)
from mipengine.controller.algorithms_specifications import AlgorithmSpecifications
from mipengine.controller.algorithms_specifications import InputDataStatType
from mipengine.controller.algorithms_specifications import InputDataType
from mipengine.controller.algorithms_specifications import ParameterSpecification
from mipengine.controller.algorithms_specifications import InputDataSpecification
from mipengine.controller.algorithms_specifications import InputDataSpecifications
from mipengine.controller.algorithms_specifications import algorithms_specifications
from mipengine.controller.api.algorithm_request_dto import AlgorithmInputDataDTO
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO

from mipengine.controller.api.exceptions import BadRequest
from mipengine.controller.api.exceptions import BadUserInput

from mipengine.controller import config
from mipengine.filters import validate_filter


# TODO This validator will be refactored heavily with https://team-1617704806227.atlassian.net/browse/MIP-90


def validate_algorithm_request(
    algorithm_name: str,
    algorithm_request_dto: AlgorithmRequestDTO,
    available_datasets_per_schema: Dict[str, List[str]],
):
    algorithm_specs = _get_algorithm_specs(algorithm_name)
    _validate_algorithm_request_body(
        algorithm_request_dto=algorithm_request_dto,
        algorithm_specs=algorithm_specs,
        available_datasets_per_schema=available_datasets_per_schema,
    )


def _get_algorithm_specs(algorithm_name):
    if algorithm_name not in algorithms_specifications.enabled_algorithms.keys():
        raise BadRequest(f"Algorithm '{algorithm_name}' does not exist.")
    return algorithms_specifications.enabled_algorithms[algorithm_name]


def _validate_algorithm_request_body(
    algorithm_request_dto: AlgorithmRequestDTO,
    algorithm_specs: AlgorithmSpecifications,
    available_datasets_per_schema: Dict[str, List[str]],
):
    _validate_inputdata(
        inputdata=algorithm_request_dto.inputdata,
        inputdata_specs=algorithm_specs.inputdata,
        available_datasets_per_schema=available_datasets_per_schema,
    )

    _validate_parameters(
        algorithm_request_dto.parameters,
        algorithm_specs.parameters,
    )


def _validate_inputdata(
    inputdata: AlgorithmInputDataDTO,
    inputdata_specs: InputDataSpecifications,
    available_datasets_per_schema: Dict[str, List[str]],
):
    _validate_inputdata_pathology_and_dataset(
        requested_pathology=inputdata.pathology,
        requested_datasets=inputdata.datasets,
        available_datasets_per_schema=available_datasets_per_schema,
    )

    _validate_inputdata_filter(inputdata.pathology, inputdata.filters)

    _validate_algorithm_inputdatas(inputdata, inputdata_specs)


def _validate_inputdata_pathology_and_dataset(
    requested_pathology: str,
    requested_datasets: List[str],
    available_datasets_per_schema: Dict[str, List[str]],
):
    """
    Validates that the pathology, dataset values exist and
    that the datasets belong in the pathology.
    """

    if not requested_pathology in available_datasets_per_schema.keys():
        raise BadUserInput(f"Pathology '{requested_pathology}' does not exist.")

    non_existing_datasets = [
        dataset
        for dataset in requested_datasets
        if dataset not in available_datasets_per_schema[requested_pathology]
    ]
    if non_existing_datasets:
        raise BadUserInput(
            f"Datasets:'{non_existing_datasets}' could not be found for pathology:{requested_pathology}"
        )


def _validate_inputdata_filter(pathology, filter):
    """
    Validates that the filter provided have the correct format
    following: https://querybuilder.js.org/
    """
    common_data_elements = CommonDataElements(config.cdes_metadata_path)
    validate_filter(common_data_elements, pathology, filter)


# TODO This will be removed with the dynamic inputdata logic.
def _validate_algorithm_inputdatas(
    inputdata: AlgorithmInputDataDTO, inputdata_specs: InputDataSpecifications
):
    _validate_algorithm_inputdata(inputdata.x, inputdata_specs.x, inputdata.pathology)
    _validate_algorithm_inputdata(inputdata.y, inputdata_specs.y, inputdata.pathology)


def _validate_algorithm_inputdata(
    inputdata_values: Optional[List[str]],
    inputdata_spec: InputDataSpecification,
    pathology: str,
):
    if not inputdata_values:
        if inputdata_spec.notblank:
            raise BadUserInput(
                f"Inputdata '{inputdata_spec.label}' should be provided."
            )
        else:
            return

    _validate_inputdata_values_quantity(inputdata_values, inputdata_spec)

    for inputdata_value in inputdata_values:
        _validate_inputdata_value(inputdata_value, inputdata_spec, pathology)


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
    inputdata_value: str, inputdata_specs: InputDataSpecification, pathology: str
):
    inputdata_value_metadata = _get_cde_metadata(inputdata_value, pathology)
    _validate_inputdata_types(
        inputdata_value, inputdata_specs, inputdata_value_metadata
    )
    _validate_inputdata_stattypes(
        inputdata_value, inputdata_specs, inputdata_value_metadata
    )
    _validate_inputdata_enumerations(
        inputdata_value, inputdata_specs, inputdata_value_metadata
    )


def _get_cde_metadata(cde, pathology):
    pathology_cdes: Dict[
        str, CommonDataElement
    ] = controller_common_data_elements.pathologies[pathology]
    if cde not in pathology_cdes.keys():
        raise BadUserInput(
            f"The CDE '{cde}' does not exist in pathology '{pathology}'."
        )
    return pathology_cdes[cde]


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
