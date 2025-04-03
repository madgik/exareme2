from exareme2.controller import config as ctrl_config
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.services import set_worker_landscape_aggregator
from exareme2.controller.services.exaflow import (
    set_controller as set_exaflow_controller,
)
from exareme2.controller.services.exaflow.controller import (
    Controller as ExaFlowController,
)
from exareme2.controller.services.exareme2 import set_cleaner
from exareme2.controller.services.exareme2 import (
    set_controller as set_exareme2_controller,
)
from exareme2.controller.services.exareme2.cleaner import Cleaner
from exareme2.controller.services.exareme2.controller import (
    Controller as Exareme2Controller,
)
from exareme2.controller.services.exareme2.execution_engine import SMPCParams
from exareme2.controller.services.flower import set_controller as set_flower_controller
from exareme2.controller.services.flower import set_flower_execution_info
from exareme2.controller.services.flower.controller import (
    Controller as FlowerController,
)
from exareme2.controller.services.flower.flower_io_registry import FlowerIORegistry
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

    controller = Exareme2Controller(
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
    controller.start_cleanup_loop()
    set_exareme2_controller(controller)

    flower_execution_info = FlowerIORegistry(
        ctrl_config.flower_execution_timeout,
        ctrl_logger.get_background_service_logger(),
    )
    set_flower_execution_info(flower_execution_info)

    controller = FlowerController(
        flower_execution_info=flower_execution_info,
        worker_landscape_aggregator=worker_landscape_aggregator,
        task_timeout=ctrl_config.rabbitmq.celery_tasks_timeout,
    )
    set_flower_controller(controller)

    exaflow_controller = ExaFlowController(
        worker_landscape_aggregator=worker_landscape_aggregator,
        task_timeout=ctrl_config.rabbitmq.celery_tasks_timeout,
    )
    set_exaflow_controller(exaflow_controller)
