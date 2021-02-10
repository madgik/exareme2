import logging
import traceback

from mipengine.controller.algorithms_specifications import AlgorithmsSpecifications
from mipengine.controller.api.DTOs.AlgorithmExecutionDTOs import AlgorithmRequestDTO
from mipengine.controller.api.DTOs.AlgorithmSpecificationsDTOs import AlgorithmSpecifications
from mipengine.controller.api.errors.exceptions import BadRequest
from mipengine.controller.api.services.validate_algorithm_parameters import validate_algorithm_parameters


def run_algorithm(algorithm_name: str, request_body: str):
    # Check that algorithm exists
    if str.lower(algorithm_name) not in AlgorithmsSpecifications().enabled_algorithms.keys():
        raise BadRequest(f"Algorithm '{algorithm_name}' does not exist.")

    # Validate algorithm body has proper format
    try:
        algorithm_request = AlgorithmRequestDTO.from_json(request_body)
    except Exception:
        logging.error(f"Could not parse the algorithm request body. "
                      f"Exception: \n {traceback.format_exc()}")
        raise BadRequest(f"The algorithm body does not have the proper format.")

    # Get algorithm specification and validate the algorithm input
    algorithm_specs = AlgorithmSpecifications().algorithms_dict[algorithm_name]
    validate_algorithm_parameters(algorithm_specs, algorithm_request)

    # TODO Trigger algorithm execution on the controller
    return "Success!"
