from celery import shared_task

from exareme2.worker.exareme2.cleanup import cleanup_service


@shared_task
def cleanup(request_id: str, context_id: str):
    cleanup_service.cleanup(request_id, context_id)
