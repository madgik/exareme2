from exareme2.controller import config as ctrl_config
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.services import set_worker_landscape_aggregator
from exareme2.controller.services.exaflow import ExaflowController
from exareme2.controller.services.exaflow import set_exaflow_controller
from exareme2.controller.services.exareme2 import set_cleaner
from exareme2.controller.services.exareme2 import (
    set_controller as set_exareme2_controller,
)
from exareme2.controller.services.exareme2.cleaner import Cleaner
from exareme2.controller.services.exareme2.controller import (
    Exareme2Controller as Exareme2Controller,
)
from exareme2.controller.services.exareme2.execution_engine import SMPCParams
from exareme2.controller.services.flower import set_controller as set_flower_controller
from exareme2.controller.services.flower import set_flower_execution_info
from exareme2.controller.services.flower.flower_controller import (
    FlowerController as FlowerController,
)
from exareme2.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    WorkerLandscapeAggregator,
)
from exareme2.smpc_cluster_communication import DifferentialPrivacyParams


def start_background_services():
    worker_landscape_aggregator = WorkerLandscapeAggregator(
        logger=ctrl_logger.get_background_service_logger(),
        update_interval=ctrl_config.worker_landscape_aggregator_update_interval,
        tasks_timeout=ctrl_config.rabbitmq.celery_tasks_timeout,
        run_udf_task_timeout=ctrl_config.rabbitmq.celery_run_udf_task_timeout,
        deployment_type=ctrl_config.deployment_type,
        localworkers=ctrl_config.localworkers,
    )
    worker_landscape_aggregator.start()
    set_worker_landscape_aggregator(worker_landscape_aggregator)

    cleaner = Cleaner(
        logger=ctrl_logger.get_background_service_logger(),
        cleanup_interval=ctrl_config.cleanup.workers_cleanup_interval,
        contextid_release_timelimit=ctrl_config.cleanup.contextid_release_timelimit,
        cleanup_task_timeout=ctrl_config.rabbitmq.celery_cleanup_task_timeout,
        run_udf_task_timeout=ctrl_config.rabbitmq.celery_run_udf_task_timeout,
        contextids_cleanup_folder=ctrl_config.cleanup.contextids_cleanup_folder,
        worker_landscape_aggregator=worker_landscape_aggregator,
    )
    set_cleaner(cleaner)

    exareme2_controller = Exareme2Controller(
        worker_landscape_aggregator=worker_landscape_aggregator,
        cleaner=cleaner,
        logger=ctrl_logger.get_background_service_logger(),
        tasks_timeout=ctrl_config.rabbitmq.celery_tasks_timeout,
        run_udf_task_timeout=ctrl_config.rabbitmq.celery_run_udf_task_timeout,
        smpc_params=SMPCParams(
            smpc_enabled=ctrl_config.smpc.enabled or False,
            smpc_optional=ctrl_config.smpc.optional or False,
            dp_params=DifferentialPrivacyParams(
                sensitivity=ctrl_config.smpc.dp.sensitivity,
                privacy_budget=ctrl_config.smpc.dp.privacy_budget,
            )
            if ctrl_config.smpc.dp.enabled
            else None,
        ),
    )
    exareme2_controller.start_cleanup_loop()
    set_exareme2_controller(exareme2_controller)

    flower_controller = FlowerController(
        worker_landscape_aggregator=worker_landscape_aggregator,
        task_timeout=ctrl_config.rabbitmq.celery_tasks_timeout,
    )
    set_flower_controller(flower_controller)

    exaflow_controller = ExaflowController(
        worker_landscape_aggregator=worker_landscape_aggregator,
        task_timeout=ctrl_config.rabbitmq.celery_tasks_timeout,
    )
    set_exaflow_controller(exaflow_controller)
