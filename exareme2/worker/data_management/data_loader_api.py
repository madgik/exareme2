from celery import shared_task

from exareme2.worker.data_management import data_loader_service


@shared_task
def load_data_folder(request_id: str, folder: str | None = None) -> str:
    return data_loader_service.load_data_folder(request_id, folder)
