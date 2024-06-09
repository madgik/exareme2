from logging.config import dictConfig

import pydantic
from celery.utils.serialization import jsonify
from quart import Blueprint
from quart import request

from exareme2.controller.quart.loggers import loggers
from exareme2.controller.services import get_worker_landscape_aggregator
from exareme2.controller.services.algorithm_execution import execute_algorithm
from exareme2.controller.services.api.algorithm_request_dtos import AlgorithmRequestDTO
from exareme2.controller.services.api.algorithm_request_validator import BadRequest
from exareme2.controller.services.api.algorithm_spec_dtos import (
    algorithm_specifications_dtos,
)
from exareme2.controller.services.flower import get_flower_execution_info
from exareme2.controller.services.startup import start_background_services

algorithms = Blueprint("algorithms_endpoint", __name__)


@algorithms.before_app_serving
async def startup():
    configure_loggers()
    start_background_services()


@algorithms.route("/datasets", methods=["GET"])
async def get_datasets() -> dict:
    return get_worker_landscape_aggregator().get_all_available_datasets_per_data_model()


@algorithms.route("/datasets_locations", methods=["GET"])
async def get_datasets_locations() -> dict:
    return get_worker_landscape_aggregator().get_datasets_locations().datasets_locations


@algorithms.route("/cdes_metadata", methods=["GET"])
async def get_cdes_metadata() -> dict:
    cdes_per_data_model = get_worker_landscape_aggregator().get_cdes_per_data_model()

    return {
        data_model: {column: metadata.dict() for column, metadata in cdes.items()}
        for data_model, cdes in cdes_per_data_model.items()
    }


@algorithms.route("/data_models_metadata", methods=["GET"])
async def get_data_models_metadata() -> dict:
    data_models_metadata = get_worker_landscape_aggregator().get_data_models_metadata()
    return {
        data_model: data_model_metadata.to_dict()
        for data_model, data_model_metadata in data_models_metadata.items()
    }


@algorithms.route("/algorithms", methods=["GET"])
async def get_algorithms() -> str:
    return algorithm_specifications_dtos.json()


@algorithms.route("/wla", methods=["POST"])
async def update_wla() -> str:
    get_worker_landscape_aggregator().update()
    return ""


@algorithms.route("/healthcheck", methods=["GET"])
async def healthcheck() -> str:
    get_worker_landscape_aggregator().healthcheck()
    return ""


@algorithms.route("/algorithms/<algorithm_name>", methods=["POST"])
async def run_algorithm(algorithm_name: str) -> str:
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

    result = await execute_algorithm(algorithm_name, algorithm_request_dto)

    return result


@algorithms.route("/flower/input", methods=["GET"])
async def get_flower_input() -> dict:
    return get_flower_execution_info().get_inputdata().dict()


@algorithms.route("/flower/result", methods=["POST"])
async def set_flower_result():
    request_body = await request.json
    await get_flower_execution_info().set_result(result=request_body)

    return jsonify({"message": "Result set successfully"}), 200


def configure_loggers():
    """
    The loggers should be initialized at app startup, otherwise the configs are overwritten.
    """
    dictConfig(loggers)
