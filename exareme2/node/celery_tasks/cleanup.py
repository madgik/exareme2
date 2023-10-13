from celery import shared_task

import exareme2.node.services.in_database.cleanup as cleanup_service


@shared_task
def cleanup(request_id: str, context_id: str):
    cleanup_service.cleanup(request_id, context_id)
