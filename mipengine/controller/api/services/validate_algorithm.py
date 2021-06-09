import logging
import traceback
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from mipengine.common.node_catalog import node_catalog
from mipengine.controller.api.DTOs.AlgorithmRequestDTO import AlgorithmInputDataDTO
from mipengine.controller.api.DTOs.AlgorithmRequestDTO import AlgorithmRequestDTO
from mipengine.controller.api.DTOs.AlgorithmSpecificationsDTOs import (
    AlgorithmSpecificationDTO,
)
from mipengine.controller.api.DTOs.AlgorithmSpecificationsDTOs import (
    algorithm_specificationsDTOs,
)
from mipengine.controller.api.DTOs.AlgorithmSpecificationsDTOs import (
    CrossValidationSpecificationsDTO,
)
from mipengine.controller.api.DTOs.AlgorithmSpecificationsDTOs import (
    InputDataSpecificationDTO,
)
from mipengine.controller.api.DTOs.AlgorithmSpecificationsDTOs import (
    InputDataSpecificationsDTO,
)
from mipengine.controller.api.errors.exceptions import BadRequest
from mipengine.controller.api.errors.exceptions import BadUserInput
from mipengine.controller.algorithms_specifications import GenericParameterSpecification
from mipengine.common.common_data_elements import CommonDataElement
from mipengine.common.common_data_elements import common_data_elements


def validate_algorithm(algorithm_name: str, request_body: str):
    """
    Validates the proper usage of the algorithm:
    1) algorithm exists,
    2) algorithm body has proper format and
    3) algorithm input matches the algorithm specifications.
    """

    # Check that algorithm exists
    if (
        str.lower(algorithm_name)
        not in algorithm_specificationsDTOs.algorithms_dict.keys()
    ):
        raise BadRequest(f"Algorithm '{algorithm_name}' does not exist.")

    # Validate algorithm body has proper format
    try:
        algorithm_request = AlgorithmRequestDTO.from_json(request_body)
    except Exception:
        logging.error(
            f"Could not parse the algorithm request body. "
            f"Exception: \n {traceback.format_exc()}"
        )
        raise BadRequest(f"The algorithm body does not have the proper format.")

    # Get algorithm specification and validate the algorithm input
    algorithm_specs = algorithm_specificationsDTOs.algorithms_dict[algorithm_name]
    validate_algorithm_parameters(algorithm_specs, algorithm_request)


def validate_algorithm_parameters(
    algorithm_specs: AlgorithmSpecificationDTO, algorithm_request: AlgorithmRequestDTO
):
    # Validate inputdata
    validate_inputdata(algorithm_specs.inputdata, algorithm_request.inputdata)

    # Validate generic parameters
    validate_generic_parameters(
        algorithm_specs.parameters, algorithm_request.parameters
    )

    # Validate crossvalidation parameters
    validate_crossvalidation_parameters(
        algorithm_specs.crossvalidation, algorithm_request.crossvalidation
    )


def validate_inputdata(
    inputdata_specs: InputDataSpecificationsDTO, input_data: AlgorithmInputDataDTO
):
    """
    Validates that the:
    1) datasets/pathology exist,
    2) datasets belong in the pathology,
    3) filter have proper format
    4) and the cdes parameters have proper values.

    Validates that the algorithm's input data follow the specs.

    """
    validate_inputdata_pathology_and_dataset_values(
        input_data.pathology, input_data.datasets
    )

    validate_inputdata_filter(input_data.filters)

    validate_inputdata_cdes(inputdata_specs, input_data)


def validate_inputdata_pathology_and_dataset_values(
    pathology: str, datasets: List[str]
):
    """
    Validates that the pathology, dataset values exists and
    that the datasets belong in the pathology.
    """

    if not node_catalog.pathology_exists(pathology):
        raise BadUserInput(f"Pathology '{pathology}' does not exist.")

    if type(datasets) is not list:
        raise BadRequest(f"Datasets parameter should be a list.")

    if not all(node_catalog.dataset_exists(pathology, dataset) for dataset in datasets):
        raise BadUserInput(
            f"Datasets '{datasets}' do not belong in pathology '{pathology}'."
        )


def validate_inputdata_filter(filter):
    """
    Validates that the filter provided have the correct format
    following: https://querybuilder.js.org/
    """
    # TODO Add filter
    pass


def validate_inputdata_cdes(
    input_data_specs: InputDataSpecificationsDTO, input_data: AlgorithmInputDataDTO
):
    """
    Validates that the cdes input data (x,y) follow the specs provided
    in the algorithm specifications.
    """

    validate_inputdata_cde(input_data_specs.x, input_data.x, input_data.pathology)
    validate_inputdata_cde(input_data_specs.y, input_data.y, input_data.pathology)


def validate_inputdata_cde(
    cde_parameter_specs: InputDataSpecificationDTO,
    cde_parameter_value: Optional[List[str]],
    pathology: str,
):
    """
    Validates that the cde, x or y,  follows the specs provided
    in the algorithm specification.
    """

    # Validate that the cde parameters were provided, if required.
    if cde_parameter_specs.notblank and cde_parameter_value is None:
        raise BadUserInput(
            f"Inputdata '{cde_parameter_specs.label}' should be provided."
        )

    # Continue if the cde parameter was not provided
    if cde_parameter_value is None:
        return

    validate_inputdata_cdes_length(cde_parameter_value, cde_parameter_specs)

    for cde in cde_parameter_value:
        validate_inputdata_cde_value(cde, cde_parameter_specs, pathology)


def validate_inputdata_cdes_length(
    cde_parameter_value: Any, cde_parameter_specs: InputDataSpecificationDTO
):
    """
    Validate that the cde inputdata has proper list length
    """
    if type(cde_parameter_value) is not list:
        raise BadRequest(f"Inputdata '{cde_parameter_specs.label}' should be a list.")

    if not cde_parameter_specs.multiple and len(cde_parameter_value) > 1:
        raise BadUserInput(
            f"Inputdata '{cde_parameter_specs.label}' cannot have multiple values."
        )


def validate_inputdata_cde_value(
    cde: str, cde_parameter_specs: InputDataSpecificationDTO, pathology: str
):
    """
    Validation of a specific cde in a parameter for the following:
    1) that it exists in the pathology CDEs,
    2) that it has a type allowed from the specification,
    3) that it has a statistical type allowed from the specification and
    4) that it has the proper amount of enumerations.
    """
    pathology_cdes: Dict[str, CommonDataElement] = common_data_elements.pathologies[
        pathology
    ]
    if cde not in pathology_cdes.keys():
        raise BadUserInput(
            f"The CDE '{cde}', of inputdata '{cde_parameter_specs.label}', "
            f"does not exist in pathology '{pathology}'."
        )

    cde_metadata: CommonDataElement = pathology_cdes[cde]
    validate_inputdata_cde_types(cde, cde_metadata, cde_parameter_specs)
    validate_inputdata_cde_stattypes(cde, cde_metadata, cde_parameter_specs)
    validate_inputdata_cde_enumerations(cde, cde_metadata, cde_parameter_specs)


def validate_inputdata_cde_types(
    cde: str,
    cde_metadata: CommonDataElement,
    cde_parameter_specs: InputDataSpecificationDTO,
):
    # Validate that the cde belongs in the allowed types
    if cde_metadata.sql_type not in cde_parameter_specs.types:
        # If "real" is allowed, "int" is allowed as well
        if cde_metadata.sql_type != "int" or "real" not in cde_parameter_specs.types:
            raise BadUserInput(
                f"The CDE '{cde}', of inputdata '{cde_parameter_specs.label}', "
                f"doesn't have one of the allowed types "
                f"'{cde_parameter_specs.types}'."
            )


def validate_inputdata_cde_stattypes(
    cde: str,
    cde_metadata: CommonDataElement,
    cde_parameter_specs: InputDataSpecificationDTO,
):
    if cde_metadata.categorical and "nominal" not in cde_parameter_specs.stattypes:
        raise BadUserInput(
            f"The CDE '{cde}', of inputdata '{cde_parameter_specs.label}', "
            f"should be categorical."
        )

    if (
        not cde_metadata.categorical
        and "numerical" not in cde_parameter_specs.stattypes
    ):
        raise BadUserInput(
            f"The CDE '{cde}', of inputdata '{cde_parameter_specs.label}', "
            f"should NOT be categorical."
        )


def validate_inputdata_cde_enumerations(
    cde: str,
    cde_metadata: CommonDataElement,
    cde_parameter_specs: InputDataSpecificationDTO,
):
    if (
        cde_parameter_specs.enumslen is not None
        and cde_parameter_specs.enumslen != len(cde_metadata.enumerations)
    ):
        raise BadUserInput(
            f"The CDE '{cde}', of inputdata '{cde_parameter_specs.label}', "
            f"should have {cde_parameter_specs.enumslen} enumerations."
        )


def validate_generic_parameters(
    parameters_specs: Optional[Dict[str, GenericParameterSpecification]],
    parameters: Optional[Dict[str, Any]],
):
    """
    If the algorithm has generic parameters (parameters),
    it validates that they follow the algorithm specs.
    """
    if parameters_specs is None:
        return

    # Validating that the parameters match with the notblank spec.
    for parameter_name, parameter_spec in parameters_specs.items():
        if not parameter_spec.notblank:
            continue
        if not parameters:
            raise BadRequest(f"Algorithm parameters not provided.")
        if parameter_name not in parameters.keys():
            raise BadUserInput(f"Parameter '{parameter_name}' should not be blank.")

        parameter_value = parameters[parameter_name]
        validate_generic_parameter_values(
            parameter_name,
            parameter_value,
            parameter_spec.type,
            parameter_spec.enums,
            parameter_spec.min,
            parameter_spec.max,
            parameter_spec.multiple,
        )


def validate_generic_parameter_values(
    parameter_name: str,
    parameter_value: Any,
    parameter_type: str,
    parameter_enums: Optional[List[Any]],
    parameter_min_value: Optional[int],
    parameter_max_value: Optional[int],
    multiple_allowed: bool,
):
    """
    Validates that the parameter value follows the specs and :
    1) has proper type,
    2) has proper enumerations, if any,
    3) is inside the min-max limits, if any,
    4) and follows the multiple values rule.
    """
    if multiple_allowed and not isinstance(parameter_value, list):
        raise BadUserInput(f"Parameter '{parameter_name}' should be a list.")

    # If the parameter value is a list, check each elements
    if multiple_allowed:
        for element in parameter_value:
            validate_generic_parameter_type(parameter_name, element, parameter_type)

            validate_generic_parameter_enumerations(
                parameter_name, element, parameter_enums
            )

            validate_generic_parameter_inside_min_max(
                parameter_name, element, parameter_min_value, parameter_max_value
            )
    else:
        validate_generic_parameter_type(parameter_name, parameter_value, parameter_type)

        validate_generic_parameter_enumerations(
            parameter_name, parameter_value, parameter_enums
        )

        validate_generic_parameter_inside_min_max(
            parameter_name, parameter_value, parameter_min_value, parameter_max_value
        )


def validate_generic_parameter_type(
    parameter_name: str, parameter_value: Any, parameter_type: str
):
    mip_types_to_python_types = {
        "text": [str],
        "int": [int],
        "real": [float, int],
        "jsonObject": [dict],
    }
    if type(parameter_value) not in mip_types_to_python_types[parameter_type]:
        raise BadUserInput(
            f"Parameter '{parameter_name}' values should be of type '{parameter_type}'."
        )


def validate_generic_parameter_enumerations(
    parameter_name: str, parameter_value: Any, parameter_enums: Optional[List[Any]]
):
    if parameter_enums is None:
        return

    if parameter_value not in parameter_enums:
        raise BadUserInput(
            f"Parameter '{parameter_name}' values "
            f"should be one of the following: '{str(parameter_enums)}'."
        )


def validate_generic_parameter_inside_min_max(
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
            f"should be lower than {parameter_max_value} ."
        )


def validate_crossvalidation_parameters(
    crossvalidation_specs: Optional[CrossValidationSpecificationsDTO],
    crossvalidation: Optional[Dict[str, Any]],
):
    """
    If crossvalidation is enabled, it validates that the algorithm's
    crossvalidation parameters follow the specs.
    """

    # Cross validation is optional, if not present, do nothing
    if crossvalidation_specs is None or crossvalidation is None:
        return

    validate_generic_parameters(crossvalidation_specs.parameters, crossvalidation)
