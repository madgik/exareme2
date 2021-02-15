from celery import Celery

from mipengine.node.config.config_parser import Config

config = Config().config
ip = config["rabbitmq"]["ip"]
port = config["rabbitmq"]["port"]
user = config["rabbitmq"]["user"]
password = config["rabbitmq"]["password"]
vhost = config["rabbitmq"]["vhost"]

app = Celery('mipengine.node',
             broker=f'amqp://{user}:{password}@{ip}:{port}/{vhost}',
             backend='rpc://',
             include=['mipengine.node.tasks.tables', 'mipengine.node.tasks.remote_tables', 'mipengine.node.tasks.merge_tables', 'mipengine.node.tasks.views', 'mipengine.node.tasks.common'])
