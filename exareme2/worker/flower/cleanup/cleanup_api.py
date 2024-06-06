from celery import shared_task

from exareme2.worker.flower.cleanup import cleanup_service


@shared_task
def stop_flower_server(request_id: str, pid: int, algorithm_name: str):
    cleanup_service.stop_flower_process(request_id, pid, algorithm_name)


@shared_task
def stop_flower_client(request_id: str, pid: int, algorithm_name: str):
    cleanup_service.stop_flower_process(request_id, pid, algorithm_name)


@shared_task
def garbage_collect(request_id: str):
    cleanup_service.garbage_collect(request_id)
