from celery import shared_task

from exareme2.worker.worker_info import worker_info_service


@shared_task
def get_worker_info(request_id: str) -> str:
    return worker_info_service.get_worker_info(request_id).json()


@shared_task
def get_worker_datasets_per_data_model(request_id: str) -> str:
    return worker_info_service.get_worker_datasets_per_data_model(request_id).json()


@shared_task
def get_data_model_attributes(request_id: str, data_model: str) -> str:
    return worker_info_service.get_data_model_attributes(request_id, data_model).json()


@shared_task
def get_data_model_cdes(request_id: str, data_model: str) -> str:
    return worker_info_service.get_data_model_cdes(request_id, data_model).json()


@shared_task
def healthcheck(request_id: str, check_db):
    return worker_info_service.healthcheck(request_id, check_db)
