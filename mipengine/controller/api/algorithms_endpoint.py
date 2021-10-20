import pydantic
from quart import Blueprint
from quart import request

from mipengine.controller.api.algorithm_request_dto import (
    AlgorithmRequestDTO,
)
from mipengine.controller.api.algorithm_specifications_dtos import (
    AlgorithmSpecificationDTO,
)
from mipengine.controller.api.algorithm_specifications_dtos import (
    algorithm_specificationsDTOs,
)
from mipengine.controller.api.exceptions import BadRequest
from mipengine.controller.controller import Controller
from mipengine.node_tasks_DTOs import PrivacyError

algorithms = Blueprint("algorithms_endpoint", __name__)
controller = Controller()


@algorithms.before_app_serving
async def startup():
    await controller.start_node_registry()


@algorithms.after_app_serving
async def shutdown():
    await controller.stop_node_registry()


@algorithms.route("/datasets", methods=["GET"])
async def get_datasets() -> dict:
    return controller.get_all_datasets_per_node()


@algorithms.route("/algorithms", methods=["GET"])
async def get_algorithms() -> str:
    algorithm_specifications = algorithm_specificationsDTOs.algorithms_list

    return AlgorithmSpecificationDTO.schema().dumps(algorithm_specifications, many=True)


@algorithms.route("/algorithms/<algorithm_name>", methods=["POST"])
async def post_algorithm(algorithm_name: str) -> str:
    try:
        request_body = await request.json
        algorithm_request_dto = AlgorithmRequestDTO.parse_obj(request_body)
    except pydantic.error_wrappers.ValidationError as pydantic_error:
        error_msg = (
            f"Algorithm execution request malformed:"
            f"\nrequest received:{request_body}"
            f"\nerror:{pydantic_error}"
        )
        raise BadRequest(error_msg)

    controller.validate_algorithm_execution_request(
        algorithm_name=algorithm_name, algorithm_request_dto=algorithm_request_dto
    )

    algorithm_result = await controller.exec_algorithm(
        algorithm_name=algorithm_name, algorithm_request_dto=algorithm_request_dto
    )

    return algorithm_result
