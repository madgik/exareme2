import traceback
import asyncio
import pydantic

from quart import Blueprint
from quart import request

from werkzeug.exceptions import BadRequest

from mipengine.controller.api.exceptions import BadRequest
from mipengine.controller.algorithm_execution_DTOs import AlgorithmRequestDTO
from mipengine.controller.controller import Controller
from mipengine.controller.api.exceptions import BadUserInput
from mipengine.controller.api.exceptions import UnexpectedException
from mipengine.controller.api.validator import validate_algorithm_request

from mipengine.controller.api.algorithm_specifications_dtos import (
    AlgorithmSpecificationDTO,
)
from mipengine.controller.api.algorithm_specifications_dtos import (
    algorithm_specificationsDTOs,
)
algorithms = Blueprint("algorithms_endpoint", __name__)
controller = Controller()


@algorithms.before_app_serving
async def startup():
    await controller.start_node_registry()


@algorithms.after_app_serving
async def shutdown():
    await controller.stop_node_registry()


@algorithms.route("/datasets")
async def get_datasets() -> dict:
    return controller.get_all_datasets_per_node()

@algorithms.route("/algorithms")  # TODO methods=["GET"]
async def get_algorithms() -> str:
    algorithm_specifications = algorithm_specificationsDTOs.algorithms_list

    return AlgorithmSpecificationDTO.schema().dumps(algorithm_specifications, many=True)


@algorithms.route("/algorithms/<algorithm_name>", methods=["POST"])
async def post_algorithm(algorithm_name: str) -> str:

    #DEBUG(future logging..)
    print(
        f"(algorithm_endpoints.py::post_algorithm) just received request for  executing->  {algorithm_name=}"
    )
    #DEBUG end

    #1. Parse the request body to ALgorithmRequestDTO
    algorithm_request_dto = None
    try:
        request_body = await request.json
        algorithm_request_dto = AlgorithmRequestDTO.parse_obj(request_body)
    except BadRequest as err:
        request_body = await request.data
        error_msg = f"{err.description=}\n The request body was:{request_body=}"
        print(error_msg)
        return error_msg
    except pydantic.error_wrappers.ValidationError as pydantic_error:
        error_msg = f"Pydantic error: {pydantic_error=}"
        print(error_msg)
        return error_msg

    #2. Validate the request
    try:
        validate_algorithm_request(
            algorithm_name=algorithm_name, algorithm_request_dto=algorithm_request_dto
        )
    except (BadRequest, BadUserInput) as exc:
        error_msg = f"\nAlgorithm request validation FAILED: {exc=}\n"
        print(error_msg)
        return error_msg
    except:
        error_msg = f"Algorithm validation failed. Exception stack trace: \n {traceback.format_exc()}"
        print(error_msg)
        return error_msg

    #DEBUG
    # ..for printing the full algorithm_request_dto object
    # from devtools import debug
    # debug(algorithm_request_dto)
    #DEBUG end

    #3. Excute the requested Algorithm
    algorithm_result = await controller.exec_algorithm(
        algorithm_name=algorithm_name, algorithm_request_dto=algorithm_request_dto
    )
    return algorithm_result
