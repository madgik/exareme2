from typing import Dict

from celery import shared_task

import exareme2.node.services.node_info as node_info_service


@shared_task
def get_node_info(request_id: str) -> str:
    return node_info_service.get_node_info(request_id).json()


@shared_task
def get_node_datasets_per_data_model(request_id: str) -> Dict[str, Dict[str, str]]:
    return node_info_service.get_node_datasets_per_data_model(request_id)


@shared_task
def get_data_model_attributes(request_id: str, data_model: str) -> str:
    return node_info_service.get_data_model_attributes(request_id, data_model).json()


@shared_task
def get_data_model_cdes(request_id: str, data_model: str) -> str:
    return node_info_service.get_data_model_cdes(request_id, data_model).json()


@shared_task
def healthcheck(request_id: str, check_db):
    return node_info_service.healthcheck(request_id, check_db)
