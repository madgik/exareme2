from celery import shared_task

from exareme2.worker.exaflow.udf import udf_service


@shared_task
def run_udf(
    request_id,
    udf_name: str,
    params: dict,
):
    return udf_service.run_udf(request_id, udf_name, params)


@shared_task
def run_monetdb_udf(
    request_id,
    udf_name: str,
    params: dict,
):
    return udf_service.run_monetdb_udf(request_id, udf_name, params)
