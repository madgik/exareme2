import os

from celery import Celery

port = os.environ['CELERY_BROKER_PORT']
app = Celery('worker',
             broker=f'amqp://user:password@localhost:{port}/user_vhost',
             backend='rpc://',
             include=['worker.tasks.tables'])
