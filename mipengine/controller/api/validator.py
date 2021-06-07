import logging
import numbers
import traceback
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from mipengine.common.common_data_elements import CommonDataElement
from mipengine.common.common_data_elements import common_data_elements
from mipengine.common.node_catalog import node_catalog
from mipengine.controller.algorithms_specifications import AlgorithmSpecifications
from mipengine.controller.algorithms_specifications import InputDataStatType
from mipengine.controller.algorithms_specifications import InputDataType
from mipengine.controller.algorithms_specifications import GenericParameterSpecification
from mipengine.controller.algorithms_specifications import InputDataSpecification
from mipengine.controller.algorithms_specifications import InputDataSpecifications
from mipengine.controller.algorithms_specifications import algorithms_specifications
from mipengine.controller.api.AlgorithmRequestDTO import AlgorithmInputDataDTO
from mipengine.controller.api.AlgorithmRequestDTO import AlgorithmRequestDTO
from mipengine.controller.api.exceptions import BadRequest
from mipengine.controller.api.exceptions import BadUserInput


# TODO This validator will be refactored heavily with https://team-1617704806227.atlassian.net/browse/MIP-68


def validate_algorithm_request(algorithm_name: str, request_body: str):

    # Validate proper algorithm request body
    # TODO Should be removed with pydantic
    try:
        algorithm_request = AlgorithmRequestDTO.from_json(request_body)
    except Exception:
        logging.error(
            f"Could not parse the algorithm request body. "
            f"Exception: \n {traceback.format_exc()}"
        )
        raise BadRequest(f"The algorithm request body does not have the proper format.")

    algorithm_specs = _get_algorithm_specs(algorithm_name)
    _validate_algorithm_parameters(algorithm_specs, algorithm_request)


def _get_algorithm_specs(algorithm_name):
    if algorithm_name not in algorithms_specifications.enabled_algorithms.keys():
        raise BadRequest(f"Algorithm '{algorithm_name}' does not exist.")
    return algorithms_specifications.enabled_algorithms[algorithm_name]


def _validate_algorithm_parameters(
    algorithm_specs: AlgorithmSpecifications, algorithm_request: AlgorithmRequestDTO
):
    _validate_inputdata(algorithm_specs.inputdata, algorithm_request.inputdata)

    _validate_generic_parameters(
        algorithm_specs.parameters, algorithm_request.parameters
    )


def _validate_inputdata(
    inputdata_specs: InputDataSpecifications, input_data: AlgorithmInputDataDTO
):
    _validate_pathology_and_dataset_values(input_data.pathology, input_data.datasets)

    _validate_inputdata_filter(input_data.filters)

    _validate_inputdata_cdes(inputdata_specs, input_data)


def _validate_pathology_and_dataset_values(pathology: str, datasets: List[str]):
    """
    Validates that the pathology, dataset values exist and
    that the datasets belong in the pathology.
    """

    if not node_catalog.pathology_exists(pathology):
        raise BadUserInput(f"Pathology '{pathology}' does not exist.")

    # TODO Remove with pydantic
    if not isinstance(datasets, list):
        raise BadRequest(f"Datasets parameter should be a list.")

    if not all(node_catalog.dataset_exists(pathology, dataset) for dataset in datasets):
        raise BadUserInput(
            f"Datasets '{datasets}' do not belong in pathology '{pathology}'."
        )


def _validate_inputdata_filter(filter):
    """
    Validates that the filter provided have the correct format
    following: https://querybuilder.js.org/
    """
    # TODO Add filter
    pass


def _validate_inputdata_cdes(
    input_data_specs: InputDataSpecifications, input_data: AlgorithmInputDataDTO
):
    _validate_inputdata_cde(input_data_specs.x, input_data.x, input_data.pathology)
    _validate_inputdata_cde(input_data_specs.y, input_data.y, input_data.pathology)


def _validate_inputdata_cde(
    cde_parameter_specs: InputDataSpecification,
    cde_parameter_value: Optional[List[str]],
    pathology: str,
):
    if cde_parameter_specs.notblank and not cde_parameter_value:
        raise BadUserInput(
            f"Inputdata '{cde_parameter_specs.label}' should be provided."
        )

    if not cde_parameter_value:
        return

    _validate_inputdata_cdes_length(cde_parameter_value, cde_parameter_specs)

    for cde in cde_parameter_value:
        _validate_inputdata_cde_value(cde, cde_parameter_specs, pathology)


def _validate_inputdata_cdes_length(
    cde_parameter_value: Any, cde_parameter_specs: InputDataSpecification
):
    if not isinstance(cde_parameter_value, list):
        raise BadRequest(f"Inputdata '{cde_parameter_specs.label}' should be a list.")

    if not cde_parameter_specs.multiple and len(cde_parameter_value) > 1:
        raise BadUserInput(
            f"Inputdata '{cde_parameter_specs.label}' cannot have multiple values."
        )


def _validate_inputdata_cde_value(
    cde: str, cde_parameter_specs: InputDataSpecification, pathology: str
):
    cde_metadata = _get_pathology_cde(cde, cde_parameter_specs, pathology)
    _validate_inputdata_cde_types(cde, cde_metadata, cde_parameter_specs)
    _validate_inputdata_cde_stattypes(cde, cde_metadata, cde_parameter_specs)
    _validate_inputdata_cde_enumerations(cde, cde_metadata, cde_parameter_specs)


def _get_pathology_cde(cde, cde_parameter_specs, pathology):
    pathology_cdes: Dict[str, CommonDataElement] = common_data_elements.pathologies[
        pathology
    ]
    if cde not in pathology_cdes.keys():
        raise BadUserInput(
            f"The CDE '{cde}', of inputdata '{cde_parameter_specs.label}', "
            f"does not exist in pathology '{pathology}'."
        )
    return pathology_cdes[cde]


def _validate_inputdata_cde_types(
    cde: str,
    cde_metadata: CommonDataElement,
    cde_parameter_specs: InputDataSpecification,
):
    dtype = cde_metadata.sql_type
    dtypes = cde_parameter_specs.types
    if dtype in dtypes:
        return
    if InputDataType.REAL in dtypes and dtype in (
        InputDataType.INT,
        InputDataType.REAL,
    ):
        return
    raise BadUserInput(
        f"The CDE '{cde}', of inputdata '{cde_parameter_specs.label}', "
        f"doesn't have one of the allowed types "
        f"'{cde_parameter_specs.types}'."
    )


def _validate_inputdata_cde_stattypes(
    cde: str,
    cde_metadata: CommonDataElement,
    cde_parameter_specs: InputDataSpecification,
):
    can_be_numerical = InputDataStatType.NUMERICAL in cde_parameter_specs.stattypes
    can_be_nominal = InputDataStatType.NOMINAL in cde_parameter_specs.stattypes
    if not cde_metadata.is_categorical and not can_be_numerical:
        raise BadUserInput(
            f"The CDE '{cde}', of inputdata '{cde_parameter_specs.label}', "
            f"should be categorical."
        )
    if cde_metadata.is_categorical and not can_be_nominal:
        raise BadUserInput(
            f"The CDE '{cde}', of inputdata '{cde_parameter_specs.label}', "
            f"should NOT be categorical."
        )


def _validate_inputdata_cde_enumerations(
    cde: str,
    cde_metadata: CommonDataElement,
    cde_parameter_specs: InputDataSpecification,
):
    if (
        cde_parameter_specs.enumslen is not None
        and cde_parameter_specs.enumslen != len(cde_metadata.enumerations)
    ):
        raise BadUserInput(
            f"The CDE '{cde}', of inputdata '{cde_parameter_specs.label}', "
            f"should have {cde_parameter_specs.enumslen} enumerations."
        )


def _validate_generic_parameters(
    parameters_specs: Optional[Dict[str, GenericParameterSpecification]],
    parameters: Optional[Dict[str, Any]],
):
    """
    If the algorithm has generic parameters (parameters),
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

        parameter_value = parameters.get(parameter_name)
        if parameter_value:
            _validate_generic_parameter_values(
                parameter_name,
                parameter_value,
                parameter_spec.type,
                parameter_spec.enums,
                parameter_spec.min,
                parameter_spec.max,
                parameter_spec.multiple,
            )


def _validate_generic_parameter_values(
    parameter_name: str,
    parameter_value: Any,
    parameter_type: str,
    parameter_enums: Optional[List[Any]],
    parameter_min_value: Optional[int],
    parameter_max_value: Optional[int],
    multiple_allowed: bool,
):
    if multiple_allowed and not isinstance(parameter_value, list):
        raise BadUserInput(f"Parameter '{parameter_name}' should be a list.")

    if not multiple_allowed:
        parameter_value = [parameter_value]
    for element in parameter_value:
        _validate_generic_parameter_type(parameter_name, element, parameter_type)

        _validate_generic_parameter_enumerations(
            parameter_name, element, parameter_enums
        )

        _validate_generic_parameter_inside_min_max(
            parameter_name, element, parameter_min_value, parameter_max_value
        )


def _validate_generic_parameter_type(
    parameter_name: str, parameter_value: Any, parameter_type: str
):
    mip_types_to_python_types = {
        "text": str,
        "int": int,
        "real": numbers.Real,
    }
    if not isinstance(parameter_value, mip_types_to_python_types[parameter_type]):
        raise BadUserInput(
            f"Parameter '{parameter_name}' values should be of type '{parameter_type}'."
        )


def _validate_generic_parameter_enumerations(
    parameter_name: str, parameter_value: Any, parameter_enums: Optional[List[Any]]
):
    if parameter_enums is None:
        return

    if parameter_value not in parameter_enums:
        raise BadUserInput(
            f"Parameter '{parameter_name}' values "
            f"should be one of the following: '{str(parameter_enums)}'."
        )


def _validate_generic_parameter_inside_min_max(
    parameter_name: str,
    parameter_value: Any,
    parameter_min_value: Optional[int],
    parameter_max_value: Optional[int],
):
    if parameter_min_value is None and parameter_max_value is None:
        return

    if parameter_min_value is not None and parameter_value < parameter_min_value:
        raise BadUserInput(
            f"Parameter '{parameter_name}' values "
            f"should be greater than {parameter_min_value} ."
        )

    if parameter_max_value is not None and parameter_value > parameter_max_value:
        raise BadUserInput(
            f"Parameter '{parameter_name}' values "
            f"should be less than {parameter_max_value} ."
        )
