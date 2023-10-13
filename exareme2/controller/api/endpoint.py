from logging.config import dictConfig

import pydantic
from quart import Blueprint
from quart import request

from exareme2.controller import config as controller_config
from exareme2.controller.algorithm_execution_engine import SMPCParams
from exareme2.controller.api.algorithm_request_dto import AlgorithmRequestDTO
from exareme2.controller.api.loggers import loggers
from exareme2.controller.api.specifications_dtos import algorithm_specifications_dtos
from exareme2.controller.api.validator import BadRequest
from exareme2.controller.cleaner import Cleaner
from exareme2.controller.cleaner import InitializationParams as CleanerInitParams
from exareme2.controller.controller import Controller
from exareme2.controller.controller import InitializationParams as ControllerInitParams
from exareme2.controller.node_landscape_aggregator import (
    InitializationParams as NodeLandscapeAggregatorInitParams,
)
from exareme2.controller.node_landscape_aggregator import NodeLandscapeAggregator
from exareme2.controller.uid_generator import UIDGenerator
from exareme2.smpc_cluster_communication import DifferentialPrivacyParams

algorithms = Blueprint("algorithms_endpoint", __name__)

node_landscape_aggregator = None
cleaner = None
controller = None


@algorithms.before_app_serving
async def startup():
    global node_landscape_aggregator, cleaner, controller
    node_landscape_aggregator = create_node_landscape_aggregator()
    cleaner = create_cleaner()
    controller = create_controller(
        cleaner=cleaner,
        node_landscape_aggregator=node_landscape_aggregator,
    )

    configure_loggers()
    controller.start_node_landscape_aggregator()
    controller.start_cleanup_loop()


@algorithms.route("/datasets", methods=["GET"])
async def get_datasets() -> dict:
    return controller.get_all_available_datasets_per_data_model()


@algorithms.route("/datasets_locations", methods=["GET"])
async def get_datasets_locations() -> dict:
    return controller.get_datasets_locations().datasets_locations


@algorithms.route("/cdes_metadata", methods=["GET"])
async def get_cdes_metadata() -> dict:
    return controller.get_cdes_per_data_model()


@algorithms.route("/data_models_attributes", methods=["GET"])
async def get_data_models_attributes() -> dict:
    return controller.get_data_models_attributes()


@algorithms.route("/algorithms", methods=["GET"])
async def get_algorithms() -> str:
    return algorithm_specifications_dtos.json()


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

    if not algorithm_request_dto.request_id:
        algorithm_request_dto.request_id = UIDGenerator().get_a_uid()

    controller.validate_algorithm_execution_request(
        algorithm_name=algorithm_name, algorithm_request_dto=algorithm_request_dto
    )

    algorithm_result = await controller.exec_algorithm(
        algorithm_name=algorithm_name,
        algorithm_request_dto=algorithm_request_dto,
    )

    return algorithm_result


@algorithms.route("/nla", methods=["POST"])
async def update_nla() -> str:
    node_landscape_aggregator.update()
    return ""


def configure_loggers():
    """
    The loggers should be initialized at app startup, otherwise the configs are overwritten.
    """
    dictConfig(loggers)


def create_node_landscape_aggregator():
    node_landscape_aggregator_init_params = NodeLandscapeAggregatorInitParams(
        node_landscape_aggregator_update_interval=controller_config.node_landscape_aggregator_update_interval,
        celery_tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
        celery_run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
        deployment_type=controller_config.deployment_type,
        localnodes=controller_config.localnodes,
    )
    return NodeLandscapeAggregator(node_landscape_aggregator_init_params)


def create_cleaner():
    cleaner_init_params = CleanerInitParams(
        cleanup_interval=controller_config.cleanup.nodes_cleanup_interval,
        contextid_release_timelimit=controller_config.cleanup.contextid_release_timelimit,
        celery_cleanup_task_timeout=controller_config.rabbitmq.celery_cleanup_task_timeout,
        celery_run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
        contextids_cleanup_folder=controller_config.cleanup.contextids_cleanup_folder,
        node_landscape_aggregator=node_landscape_aggregator,
    )
    return Cleaner(cleaner_init_params)


def create_controller(
    node_landscape_aggregator: NodeLandscapeAggregator, cleaner: Cleaner
):
    controller_init_params = ControllerInitParams(
        smpc_params=SMPCParams(
            smpc_enabled=controller_config.smpc.enabled or False,
            smpc_optional=controller_config.smpc.optional or False,
            dp_params=DifferentialPrivacyParams(
                sensitivity=controller_config.smpc.dp.sensitivity,
                privacy_budget=controller_config.smpc.dp.privacy_budget,
            )
            if controller_config.smpc.dp.enabled
            else None,
        ),
        celery_tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
        celery_run_udf_task_timeout=controller_config.rabbitmq.celery_run_udf_task_timeout,
    )
    return Controller(
        initialization_params=controller_init_params,
        cleaner=cleaner,
        node_landscape_aggregator=node_landscape_aggregator,
    )
