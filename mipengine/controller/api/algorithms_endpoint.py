import pydantic
from quart import Blueprint
from quart import request

from mipengine.controller.algorithm_specifications import algorithm_specifications
from mipengine.controller.api.algorithm_request_dto import AlgorithmRequestDTO
from mipengine.controller.api.exceptions import BadRequest
from mipengine.controller.controller import Controller
from mipengine.controller.controller import get_a_uniqueid

algorithms = Blueprint("algorithms_endpoint", __name__)
controller = Controller()


@algorithms.before_app_serving
async def startup():
    controller.start_node_landscape_aggregator()
    controller.start_cleanup_loop()


@algorithms.route("/datasets", methods=["GET"])
async def get_datasets() -> dict:
    return controller.get_all_available_datasets_per_data_model()


@algorithms.route("/dataset_locations", methods=["GET"])
async def get_dataset_locations() -> dict:
    return controller.get_dataset_locations()


@algorithms.route("/cdes_metadata", methods=["GET"])
async def get_cdes_metadata() -> dict:
    return {
        data_model: cdes.dict()
        for data_model, cdes in controller.get_cdes_per_data_model().items()
    }


@algorithms.route("/algorithms", methods=["GET"])
async def get_algorithms() -> str:
    return algorithm_specifications.get_enabled_algorithm_dtos().json()


@algorithms.route("/algorithms/<algorithm_name>", methods=["POST"])
async def post_algorithm(algorithm_name: str) -> str:
    request_body = await request.json
    try:
        algorithm_request_dto = AlgorithmRequestDTO.parse_obj(request_body)
    except pydantic.error_wrappers.ValidationError as pydantic_error:
        error_msg = (
            f"Algorithm execution request malformed:"
            f"\nrequest received:{request_body}"
            f"\nerror:{pydantic_error}"
        )
        raise BadRequest(error_msg)

    request_id = algorithm_request_dto.request_id or get_a_uniqueid()
    controller.validate_algorithm_execution_request(
        algorithm_name=algorithm_name, algorithm_request_dto=algorithm_request_dto
    )

    algorithm_result = await controller.exec_algorithm(
        request_id=request_id,
        algorithm_name=algorithm_name,
        algorithm_request_dto=algorithm_request_dto,
    )

    return algorithm_result
