from celery import Celery

from worker.config.config_parser import Config

config = Config().config
ip = config["rabbitmq"]["ip"]
port = config["rabbitmq"]["port"]
user = config["rabbitmq"]["user"]
password = config["rabbitmq"]["password"]
vhost = config["rabbitmq"]["vhost"]

app = Celery('worker',
             broker=f'amqp://{user}:{password}@{ip}:{port}/{vhost}',
             backend='rpc://',
             include=['worker.tasks.tables'])
