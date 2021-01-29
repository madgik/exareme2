from typing import List

import jsonpickle
from celery import Celery

from tasks.data_classes import ColumnInfo
from tasks.tables import TableInfo

# TODO Convert to a test framework

node1 = Celery('tables',
               broker='amqp://kostas:1234@localhost:5676/kostas_vhost',
               backend='rpc://',
               include=['tasks.tables'])
#
# node2 = Celery('tasks',
#                broker='amqp://kostas:1234@localhost:5677/kostas_vhost',
#                backend='rpc://',
#                include=['tasks.tasks'])
#
# node3 = Celery('tasks',
#                broker='amqp://kostas:1234@localhost:5678/kostas_vhost',
#                backend='rpc://',
#                include=['tasks.tasks'])

# create some tasks
# task_node1 = node1.signature('tasks.tables.create')
# task_node2 = node1.signature('tasks.tasks.delete')
# task_node2 = node2.signature('tasks.tasks.print_hello')
# task_node3 = node3.signature('tasks.tasks.print_hello')

create_node = node1.signature('tasks.tables.create_table')
delete_node = node1.signature('tasks.tables.delete_table')
get_node = node1.signature('tasks.tables.get_table_data')
getall_node = node1.signature('tasks.tables.get_tables_info')

create = create_node.delay(
    [{"column_name": "col1", "column_type": "INT"}, {"column_name": "col2", "column_type": "FLOAT"},
     {"column_name": "col3", "column_type": "TEXT"}], "regression_211232433443").get()
# table_name = TableInfo.from_json(create).name
# get_node.delay(table_name).get()
# delete_node.delay(table_name)
getall_node.delay()
