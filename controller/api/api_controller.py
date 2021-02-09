import logging
import traceback

from quart import Quart, request

from controller.api.DTOs.AlgorithmSpecificationsDTOs import AlgorithmDTO, AlgorithmSpecifications
from controller.api.errors import BadRequest, BadUserInput
from controller.api.services.run_algorithm import run_algorithm

app = Quart(__name__)


# TODO break into views/app/errors


@app.route("/algorithms")
async def get_algorithms() -> str:
    algorithm_specifications = AlgorithmSpecifications().algorithms_list

    return AlgorithmDTO.schema().dumps(algorithm_specifications, many=True)


@app.route("/algorithms/<algorithm_name>", methods=['POST'])
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


@app.errorhandler(BadRequest)
def handle_bad_request(error: BadRequest):
    return error.message, error.status_code


@app.errorhandler(BadUserInput)
def handle_bad_user_input(error: BadUserInput):
    return error.message, error.status_code
