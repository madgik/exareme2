import os

from celery import Celery

port = os.environ['CELERY_BROKER_PORT']
app = Celery('worker',
             broker=f'amqp://kostas:1234@localhost:{port}/kostas_vhost',
             backend='rpc://',
             include=['tasks.tables'])
