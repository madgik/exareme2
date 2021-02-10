import logging
import traceback

from quart import request, Blueprint

from mipengine.controller.api.DTOs.AlgorithmSpecificationsDTOs import AlgorithmSpecificationDTO, AlgorithmSpecifications
from mipengine.controller.api.errors.exceptions import BadRequest, BadUserInput
from mipengine.controller.api.services.run_algorithm import run_algorithm

algorithms = Blueprint('algorithms_endpoint', __name__)


@algorithms.route("/algorithms")
async def get_algorithms() -> str:
    algorithm_specifications = AlgorithmSpecifications().algorithms_list

    return AlgorithmSpecificationDTO.schema().dumps(algorithm_specifications, many=True)


@algorithms.route("/algorithms/<algorithm_name>", methods=['POST'])
async def post_algorithm(algorithm_name: str) -> str:
    logging.info(f"Algorithm execution with name {algorithm_name}.")

    request_body = await request.data

    try:
        response = run_algorithm(algorithm_name, request_body)
    except (BadRequest, BadUserInput) as exc:
        raise exc
    except:
        logging.error(f"Unhandled exception: \n {traceback.format_exc()}")
        raise BadRequest("Something went wrong. "
                         "Please inform the system administrator or try again later.")

    return response
