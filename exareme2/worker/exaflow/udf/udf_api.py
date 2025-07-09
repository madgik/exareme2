from celery import shared_task

from exareme2.worker.exaflow.udf import udf_service


@shared_task
def run_udf(
    request_id,
    udf_registry_key: str,
    params: dict,
):
    return udf_service.run_udf(request_id, udf_registry_key, params)
