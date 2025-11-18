from exaflow.controller import config as ctrl_config
from exaflow.controller import logger as ctrl_logger
from exaflow.controller.services import set_worker_landscape_aggregator
from exaflow.controller.services.exaflow import ExaflowController
from exaflow.controller.services.exaflow import set_exaflow_controller
from exaflow.controller.services.flower import set_flower_controller
from exaflow.controller.services.flower.controller import FlowerController
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    WorkerLandscapeAggregator,
)
from exaflow.smpc_cluster_communication import DifferentialPrivacyParams


def start_background_services():
    worker_landscape_aggregator = WorkerLandscapeAggregator(
        logger=ctrl_logger.get_background_service_logger(),
        update_interval=ctrl_config.worker_landscape_aggregator_update_interval,
        tasks_timeout=ctrl_config.grpc.tasks_timeout,
        deployment_type=ctrl_config.deployment_type,
        localworkers=ctrl_config.localworkers,
    )
    worker_landscape_aggregator.start()
    set_worker_landscape_aggregator(worker_landscape_aggregator)

    flower_controller = FlowerController(
        worker_landscape_aggregator=worker_landscape_aggregator,
        task_timeout=ctrl_config.grpc.tasks_timeout,
    )
    set_flower_controller(flower_controller)

    exaflow_controller = ExaflowController(
        worker_landscape_aggregator=worker_landscape_aggregator,
        task_timeout=ctrl_config.grpc.tasks_timeout,
    )
    set_exaflow_controller(exaflow_controller)
