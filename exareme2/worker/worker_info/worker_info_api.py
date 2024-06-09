from typing import Dict
from typing import List
from typing import Tuple

from celery import shared_task

from exareme2.worker.worker_info import worker_info_service
from exareme2.worker_communication import DataModelMetadata


@shared_task
def get_worker_info(request_id: str) -> str:
    return worker_info_service.get_worker_info(request_id).json()


@shared_task
def get_worker_data_model_metadata_and_datasets(
    request_id: str,
) -> Tuple[Dict[str, DataModelMetadata], Dict[str, List[str]]]:
    return worker_info_service.get_worker_data_model_metadata_and_datasets(request_id)


@shared_task
def healthcheck(request_id: str, check_db):
    return worker_info_service.healthcheck(request_id, check_db)
