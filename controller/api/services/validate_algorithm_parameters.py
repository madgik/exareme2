from typing import Optional, Dict, Any, List

from controller.algorithms import GenericParameter
from controller.api.DTOs.AlgorithmExecutionDTOs import AlgorithmRequestDTO
from controller.api.DTOs.AlgorithmSpecificationsDTOs import AlgorithmDTO, \
    InputDataParameterDTO, CrossValidationParametersDTO, INPUTDATA_PATHOLOGY_PARAMETER_NAME, \
    INPUTDATA_DATASET_PARAMETER_NAME, \
    INPUTDATA_FILTERS_PARAMETER_NAME, INPUTDATA_X_PARAMETER_NAME, INPUTDATA_Y_PARAMETER_NAME
from controller.api.errors import BadRequest
from controller.worker_catalogue import WorkerCatalogue


def validate_algorithm_parameters(algorithm_specs: AlgorithmDTO,
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


def validate_inputdata(inputdata_specs: Dict[str, InputDataParameterDTO],
                       input_data: Dict[str, Any]):
    """
    Validates that the:
    1) datasets/pathology exist,
    2) datasets belong in the pathology,
    3) filters have proper format
    4) and the cdes parameters have proper values.

    Validates that the algorithm's input data follow the specs.

    """
    validate_proper_pathology_and_dataset_values(input_data[INPUTDATA_PATHOLOGY_PARAMETER_NAME],
                                                 input_data[INPUTDATA_DATASET_PARAMETER_NAME])

    validate_proper_filters(input_data[INPUTDATA_FILTERS_PARAMETER_NAME])

    validate_proper_cde_values(inputdata_specs[INPUTDATA_X_PARAMETER_NAME],
                               input_data[INPUTDATA_X_PARAMETER_NAME])

    validate_proper_cde_values(inputdata_specs[INPUTDATA_Y_PARAMETER_NAME],
                               input_data[INPUTDATA_Y_PARAMETER_NAME])


def validate_proper_pathology_and_dataset_values(pathology: str,
                                                 datasets: List[str]):
    """
    Validates that the pathology, dataset values exists and
    that the datasets belong in the pathology.
    """
    # TODO Maybe change bad request to a user error???

    worker_catalogue = WorkerCatalogue()
    if pathology not in worker_catalogue.pathologies.keys():
        raise BadRequest(f"Pathology '{pathology}' does not exist.")

    if type(datasets) is not list:
        raise BadRequest(f"Datasets parameter should be a list.")

    if not all(dataset in worker_catalogue.pathologies[pathology] for dataset in datasets):
        raise BadRequest(f"Datasets '{datasets}' do not belong in pathology '{pathology}'.")


def validate_proper_filters(filters):
    """
    Validates that the filters provided have the correct format
    following: https://querybuilder.js.org/
    """
    # TODO Add filters
    pass


def validate_proper_cde_values(cde_parameter_specs: InputDataParameterDTO,
                               cde_parameter_value: List[str]):
    """
    Validates that a cde parameter follows the specs provided
    in the algorithm properties.
    """
    # TODO Next one!
    pass


def validate_generic_parameters(parameters_specs: Optional[Dict[str, GenericParameter]],
                                parameters: Optional[Dict[str, Any]]):
    """
    If the algorithm has generic parameters (parameters),
    it validates that they follow the algorithm specs.
    """
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
        validate_proper_parameter_values(parameter_name,
                                         parameter_value,
                                         parameter_spec.type,
                                         parameter_spec.enums,
                                         parameter_spec.min,
                                         parameter_spec.max,
                                         parameter_spec.multiple)


def validate_crossvalidation_parameters(crossvalidation_specs: Optional[CrossValidationParametersDTO],
                                        crossvalidation: Optional[Dict[str, Any]]):
    """
    If crossvalidation is enabled, it validates that the algorithm's
    crossvalidation parameters follow the specs.
    """

    if crossvalidation_specs is None:
        return

    if crossvalidation is None:
        raise BadRequest("Crossvalidation parameters not provided.")

    validate_generic_parameters(crossvalidation_specs.parameters,
                                crossvalidation)


def validate_proper_parameter_values(parameter_name: str,
                                     parameter_value: Any,
                                     parameter_type: str,
                                     parameter_enums: Optional[List[Any]],
                                     parameter_min_value: Optional[int],
                                     parameter_max_value: Optional[int],
                                     multiple_allowed: bool
                                     ):
    """
    Validates that the parameter value follows the specs and :
    1) has proper type,
    2) has proper enumerations, if any,
    3) is inside the min-max limits, if any,
    4) and follows the multiple values rule.
    """
    if multiple_allowed and not isinstance(parameter_value, list):
        raise BadRequest(f"Parameter '{parameter_name}' should be a list.")

    # If the parameter value is a list, check each elements
    if multiple_allowed:
        for element in parameter_value:
            validate_proper_parameter_type(parameter_name,
                                           element,
                                           parameter_type)

            validate_proper_parameter_enumerations(parameter_name,
                                                   element,
                                                   parameter_enums)

            validate_parameter_inside_min_max(parameter_name,
                                              element,
                                              parameter_min_value,
                                              parameter_max_value)
    else:
        validate_proper_parameter_type(parameter_name,
                                       parameter_value,
                                       parameter_type)

        validate_proper_parameter_enumerations(parameter_name,
                                               parameter_value,
                                               parameter_enums)

        validate_parameter_inside_min_max(parameter_name,
                                          parameter_value,
                                          parameter_min_value,
                                          parameter_max_value)


def validate_proper_parameter_type(parameter_name: str,
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


def validate_proper_parameter_enumerations(parameter_name: str,
                                           parameter_value: Any,
                                           parameter_enums: Optional[List[Any]]
                                           ):
    if parameter_enums is None:
        return

    if parameter_value not in parameter_enums:
        raise BadRequest(
            f"Parameter '{parameter_name}' values should be one of the following: '{str(parameter_enums)}' .")


def validate_parameter_inside_min_max(parameter_name: str,
                                      parameter_value: Any,
                                      parameter_min_value: Optional[int],
                                      parameter_max_value: Optional[int]
                                      ):
    if parameter_min_value is None and parameter_max_value is None:
        return

    if parameter_min_value is not None and parameter_value < parameter_min_value:
        raise BadRequest(
            f"Parameter '{parameter_name}' values should be greater than {parameter_min_value} .")

    if parameter_max_value is not None and parameter_value > parameter_max_value:
        raise BadRequest(
            f"Parameter '{parameter_name}' values should be lower than {parameter_max_value} .")
