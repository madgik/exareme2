from celery import Celery
from mipengine.config.config_parser import Config


config = Config().config
ip = config["rabbitmq"]["ip"]
port = config["rabbitmq"]["port"]
user = config["rabbitmq"]["user"]
password = config["rabbitmq"]["password"]
vhost = config["rabbitmq"]["vhost"]

app = Celery('mipengine.worker',
             broker=f'amqp://{user}:{password}@{ip}:{port}/{vhost}',
             backend='rpc://',
             include=['mipengine.worker.tasks.tables', 'mipengine.worker.tasks.remote_tables', 'mipengine.worker.tasks.merge_tables', 'mipengine.worker.tasks.views'])
