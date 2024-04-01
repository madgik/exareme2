from typing import List

from celery import shared_task

from exareme2.worker.exareme2.views import views_service


@shared_task
def get_views(request_id: str, context_id: str) -> List[str]:
    return views_service.get_views(request_id, context_id)


@shared_task
def create_data_model_views(
    request_id: str,
    context_id: str,
    command_id: str,
    data_model: str,
    datasets: List[str],
    columns_per_view: List[List[str]],
    filters: dict = None,
    dropna: bool = True,
    check_min_rows: bool = True,
) -> List[str]:
    return [
        view.json()
        for view in views_service.create_data_model_views(
            request_id,
            context_id,
            command_id,
            data_model,
            datasets,
            columns_per_view,
            filters,
            dropna,
            check_min_rows,
        )
    ]


@shared_task
def create_view(
    request_id: str,
    context_id: str,
    command_id: str,
    table_name: str,
    columns: List[str],
    filters: dict,
) -> str:
    return views_service.create_view(
        request_id, context_id, command_id, table_name, columns, filters
    ).json()
