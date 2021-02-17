from celery import shared_task

from mipengine.node.monetdb_interface import common


@shared_task
def clean_up(context_id: str):
    common.clean_up(context_id)
