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
from mipengine.controller.algorithms_specifications import ParameterSpecification
from mipengine.controller.algorithms_specifications import InputDataSpecification
from mipengine.controller.algorithms_specifications import algorithms_specifications
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO
from mipengine.controller.api.exceptions import BadRequest
from mipengine.controller.api.exceptions import BadUserInput


# TODO This validator will be refactored heavily with https://team-1617704806227.atlassian.net/browse/MIP-68


def validate_algorithm_request(algorithm_name: str, request_body: str):

    # Validate proper algorithm request body
    # TODO Should be removed with pydantic
    try:
        print(request_body)
        algorithm_request = AlgorithmRequestDTO.from_json(request_body)
    except Exception:
        logging.error(
            f"Could not parse the algorithm request body. "
            f"Exception: \n {traceback.format_exc()}"
        )
        raise BadRequest(f"The algorithm request body does not have the proper format.")

    algorithm_specs = _get_algorithm_specs(algorithm_name)
    _validate_algorithm_request_body(algorithm_request, algorithm_specs)


def _get_algorithm_specs(algorithm_name):
    if algorithm_name not in algorithms_specifications.enabled_algorithms.keys():
        raise BadRequest(f"Algorithm '{algorithm_name}' does not exist.")
    return algorithms_specifications.enabled_algorithms[algorithm_name]


def _validate_algorithm_request_body(
    algorithm_request_body: AlgorithmRequestDTO,
    algorithm_specs: AlgorithmSpecifications,
):
    _validate_inputdata(algorithm_request_body.inputdata, algorithm_specs.inputdata)

    _validate_parameters(
        algorithm_request_body.parameters,
        algorithm_specs.parameters,
    )


def _validate_inputdata(
    inputdata: Dict[str, Any],
    algorithm_inputdata_specs: Dict[str, InputDataSpecification],
):
    _validate_inputdata_pathology_and_dataset(
        inputdata.get("pathology"), inputdata.get("datasets")
    )

    _validate_inputdata_filter(inputdata.get("filter"))

    for inputdata_name, inputdata_spec in algorithm_inputdata_specs.items():
        _validate_algorithm_inputdata(
            inputdata.get(inputdata_name), inputdata_spec, inputdata.get("pathology")
        )


def _validate_inputdata_pathology_and_dataset(pathology: str, datasets: List[str]):
    """
    Validates that the pathology, dataset values exist and
    that the datasets belong in the pathology.
    """
    # TODO Refactor with pydantic

    if not pathology:
        raise BadRequest("Pathology inputdata not provided.")
    if not node_catalog.pathology_exists(pathology):
        raise BadUserInput(f"Pathology '{pathology}' does not exist.")

    if not datasets:
        raise BadRequest("Datasets inputdata not provided.")
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
    pathology_cdes: Dict[str, CommonDataElement] = common_data_elements.pathologies[
        pathology
    ]
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
