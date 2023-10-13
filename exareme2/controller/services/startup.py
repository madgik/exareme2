import exareme2.controller as controller_package
import exareme2.controller.services.in_database as in_database_package
from exareme2.controller import config as ctrl_config
from exareme2.controller import logger as ctrl_logger
from exareme2.controller.services.in_database.cleaner import Cleaner
from exareme2.controller.services.in_database.controller import Controller
from exareme2.controller.services.in_database.execution_engine import SMPCParams
from exareme2.controller.services.node_landscape_aggregator import (
    NodeLandscapeAggregator,
)
from exareme2.smpc_cluster_communication import DifferentialPrivacyParams


def start_background_services():
    controller_package.node_landscape_aggregator = NodeLandscapeAggregator(
        logger=ctrl_logger.get_background_service_logger(),
        update_interval=ctrl_config.node_landscape_aggregator_update_interval,
        tasks_timeout=ctrl_config.rabbitmq.celery_tasks_timeout,
        run_udf_task_timeout=ctrl_config.rabbitmq.celery_run_udf_task_timeout,
        deployment_type=ctrl_config.deployment_type,
        localnodes=ctrl_config.localnodes,
    )
    controller_package.node_landscape_aggregator.start()

    in_database_package.cleaner = Cleaner(
        logger=ctrl_logger.get_background_service_logger(),
        cleanup_interval=ctrl_config.cleanup.nodes_cleanup_interval,
        contextid_release_timelimit=ctrl_config.cleanup.contextid_release_timelimit,
        cleanup_task_timeout=ctrl_config.rabbitmq.celery_cleanup_task_timeout,
        run_udf_task_timeout=ctrl_config.rabbitmq.celery_run_udf_task_timeout,
        contextids_cleanup_folder=ctrl_config.cleanup.contextids_cleanup_folder,
        node_landscape_aggregator=controller_package.node_landscape_aggregator,
    )

    in_database_package.controller = Controller(
        node_landscape_aggregator=controller_package.node_landscape_aggregator,
        cleaner=in_database_package.cleaner,
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
    in_database_package.controller.start_cleanup_loop()
