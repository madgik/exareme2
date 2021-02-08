import logging
import traceback
from typing import Optional, Dict, Any, List

from controller.algorithms import Algorithms, GenericParameter
from controller.api.DTOs.AlgorithmExecutionDTOs import AlgorithmRequestDTO
from controller.api.DTOs.AlgorithmSpecificationsDTOs import AlgorithmSpecifications, AlgorithmDTO, \
    InputDataParameterDTO, CrossValidationParametersDTO
from controller.api.errors import BadRequest


def run_algorithm(algorithm_name: str, request_body: str):
    # Check that algorithm exists
    if str.lower(algorithm_name) not in Algorithms().available.keys():
        raise BadRequest(f"Algorithm '{algorithm_name}' does not exist.")

    # Validate algorithm body has proper format
    try:
        algorithm_request = AlgorithmRequestDTO.from_json(request_body)
    except Exception:
        logging.error(f"Could not parse the algorithm request body. "
                      f"Exception: \n {traceback.format_exc()}")
        raise BadRequest(f"The algorithm body does not have the proper format.")

    # Get algorithm specification and validate the request
    algorithm_specs = AlgorithmSpecifications().algorithms_dict[algorithm_name]
    validate_parameters(algorithm_specs, algorithm_request)

    # TODO Run algorithm on the controller
    pass


def validate_parameters(algorithm_specs: AlgorithmDTO,
                        algorithm_request: AlgorithmRequestDTO):
    # Validate inputdata
    validate_inputdata(algorithm_specs.inputdata,
                       algorithm_request.inputdata)

    # Validate generic parameters
    validate_generic_parameters(algorithm_specs.parameters,
                                algorithm_request.parameters)

    # Validate crossvalidation parameters
    validate_crossvalidation_parameters(algorithm_specs.crossvalidation,
                                        algorithm_request.crossvalidation)
    pass


def validate_inputdata(inputdata_specs: Dict[str, InputDataParameterDTO],
                       input_data: Dict[str, Any]):
    for inputdata_name, inputdata_spec in inputdata_specs.items():
        pass


def validate_generic_parameters(parameters_specs: Optional[Dict[str, GenericParameter]],
                                parameters: Optional[Dict[str, Any]]):
    if parameters_specs is None:
        return

    if parameters is None:
        raise BadRequest("Algorithm parameters not provided.")

    for parameter_name, parameter_spec in parameters_specs.items():
        if parameter_name not in parameters.keys():
            if parameter_spec.notblank:
                raise BadRequest(f"Parameter '{parameter_name}' should not be blank.")
            else:
                continue

        parameter_value = parameters[parameter_name]
        validate_parameter_proper_values(parameter_name,
                                         parameter_value,
                                         parameter_spec.type,
                                         parameter_spec.enums,
                                         parameter_spec.min,
                                         parameter_spec.max,
                                         parameter_spec.multiple)


def validate_crossvalidation_parameters(crossvalidation_specs: Optional[CrossValidationParametersDTO],
                                        crossvalidation: Optional[Dict[str, Any]]):
    pass


def validate_parameter_proper_values(parameter_name: str,
                                     parameter_value: Any,
                                     parameter_type: str,
                                     parameter_enums: Optional[List[Any]],
                                     parameter_min_value: Optional[int],
                                     parameter_max_value: Optional[int],
                                     multiple_allowed: bool
                                     ):
    if multiple_allowed and not isinstance(parameter_value, list):
        raise BadRequest(f"Parameter '{parameter_name}' should be a list.")

    # If the parameter value is a list, check each elements
    if multiple_allowed:
        for element in parameter_value:
            validate_parameter_proper_type(parameter_name,
                                           element,
                                           parameter_type)

            validate_parameter_proper_enumerations(parameter_name,
                                                   element,
                                                   parameter_enums)

            validate_parameter_inside_min_max(parameter_name,
                                              element,
                                              parameter_min_value,
                                              parameter_max_value)
    else:
        validate_parameter_proper_type(parameter_name,
                                       parameter_value,
                                       parameter_type)

        validate_parameter_proper_enumerations(parameter_name,
                                               parameter_value,
                                               parameter_enums)

        validate_parameter_inside_min_max(parameter_name,
                                          parameter_value,
                                          parameter_min_value,
                                          parameter_max_value)


def validate_parameter_proper_type(parameter_name: str,
                                   parameter_value: Any,
                                   parameter_type: str
                                   ):
    mip_types_to_python_types = {
        "text": [str],
        "int": [int],
        "real": [float, int],
        "jsonObject": [dict]
    }
    if type(parameter_value) not in mip_types_to_python_types[parameter_type]:
        raise BadRequest(f"Parameter '{parameter_name}' values should be of type '{parameter_type}'.")


def validate_parameter_proper_enumerations(parameter_name: str,
                                           parameter_value: Any,
                                           enumerations: Optional[List[Any]]
                                           ):
    if enumerations is None:
        return

    if parameter_value not in enumerations:
        raise BadRequest(f"Parameter '{parameter_name}' values should be one of the following: '{str(enumerations)}' .")


def validate_parameter_inside_min_max(parameter_name: str,
                                      parameter_value: Any,
                                      min: Optional[int],
                                      max: Optional[int]
                                      ):
    if min is None and max is None:
        return

    if min is not None and parameter_value < min:
        raise BadRequest(
            f"Parameter '{parameter_name}' values should be greater than {min} .")

    if max is not None and parameter_value > max:
        raise BadRequest(
            f"Parameter '{parameter_name}' values should be lower than {max} .")
