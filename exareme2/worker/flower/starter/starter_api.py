from celery import shared_task

from exareme2.worker.flower.starter import starter_service


@shared_task
def start_flower_client(
    request_id: str, algorithm_name, server_address, csv_paths, execution_timeout
) -> int:
    return starter_service.start_flower_client(
        request_id, algorithm_name, server_address, csv_paths, execution_timeout
    )


@shared_task
def start_flower_server(
    request_id: str,
    algorithm_name: str,
    number_of_clients: int,
    server_address,
    csv_paths,
) -> int:
    return starter_service.start_flower_server(
        request_id, algorithm_name, number_of_clients, server_address, csv_paths
    )
