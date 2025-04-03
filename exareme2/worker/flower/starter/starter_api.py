from celery import shared_task

from exareme2.worker.flower.starter import starter_service


@shared_task
def start_flower_client(
    request_id: str,
    algorithm_folder_path,
    server_address,
    data_model,
    datasets,
    execution_timeout,
) -> int:
    return starter_service.start_flower_client(
        request_id,
        algorithm_folder_path,
        server_address,
        data_model,
        datasets,
        execution_timeout,
    )


@shared_task
def start_flower_server(
    request_id: str,
    algorithm_folder_path: str,
    number_of_clients: int,
    server_address,
    data_model,
    datasets,
) -> int:
    return starter_service.start_flower_server(
        request_id,
        algorithm_folder_path,
        number_of_clients,
        server_address,
        data_model,
        datasets,
    )
