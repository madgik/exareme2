import time
from pprint import pprint

from celery import Celery

from mipengine.config.config_parser import Config
from mipengine.worker.monetdb_interface.common import cursor, connection
from mipengine.worker.tasks.data_classes import TableInfo

config = Config().config
ip = config["rabbitmq"]["ip"]
port = config["rabbitmq"]["port"]
user = config["rabbitmq"]["user"]
password = config["rabbitmq"]["password"]
vhost = config["rabbitmq"]["vhost"]
node1 = Celery('mipengine.worker',
               broker=f'amqp://{user}:{password}@{ip}:{port}/{vhost}',
               backend='rpc://',
               include=['mipengine.worker.tasks.tables', 'mipengine.worker.tasks.remote_tables',
                        'mipengine.worker.tasks.merge_tables'])

create_table = node1.signature('mipengine.worker.tasks.tables.create_table')
create_merge_table = node1.signature('mipengine.worker.tasks.merge_tables.create_merge_table')
clean_up = node1.signature('mipengine.worker.tasks.merge_tables.clean_up')
update_merge_table = node1.signature('mipengine.worker.tasks.merge_tables.update_merge_table')
get_merge_tables = node1.signature('mipengine.worker.tasks.merge_tables.get_merge_tables')


def setup_tables_for_merge(number_of_table: int) -> str:
    table_name = create_table.delay(f"table{number_of_table}",
                                    [{"name": "col1", "type": "INT"}, {"name": "col2", "type": "FLOAT"},
                                     {"name": "col3", "type": "TEXT"}]).get()
    connection.commit()
    cursor.execute(
        f"INSERT INTO {table_name} VALUES ( {number_of_table}, {number_of_table}, 'table_{number_of_table}' )")
    connection.commit()
    return table_name


print('Setting up tables to merge them...')

success_partition_tables = [setup_tables_for_merge(1), setup_tables_for_merge(2), setup_tables_for_merge(3),
                            setup_tables_for_merge(4)]

print('Running tests for merge tables...')

# Creating a fake table and retrieve it's data and schema afterwards.
context_id = "regression"
success_merge_table_1_name = create_merge_table.delay(context_id, success_partition_tables).get()
pprint(f"Created merge table1. Response: \n{success_merge_table_1_name}")

merge_tables = get_merge_tables.delay(context_id).get()
pprint(f"Got merge tables. Response: \n{merge_tables}")

# print('Optional empty database its important for our sanity...')
# clean_up.delay()

# In order to simulate IncompatibleSchemasMergeException
#
incompatible_table = setup_tables_for_merge(5)
cursor.execute(f"ALTER TABLE {incompatible_table} DROP {'col1'};")
connection.commit()

incompatible_partition_tables = [setup_tables_for_merge(1), setup_tables_for_merge(2), setup_tables_for_merge(3),
                                 setup_tables_for_merge(4), incompatible_table]
incompatible_merge_table_1_name = create_merge_table.delay(context_id, incompatible_partition_tables).get()
pprint(f"Created merge table1. Response: \n{incompatible_merge_table_1_name}")


# In order to simulate TableCannotBeFound uncomment the following
#
# not_found_tables = [setup_tables_for_merge(1), setup_tables_for_merge(2), setup_tables_for_merge(3),
#                     setup_tables_for_merge(4), "non_existant_table"]
#
# not_found_merge_table_1_name = create_merge_table.delay(context_id, not_found_tables).get()
# pprint(f"Created merge table1. Response: \n{not_found_merge_table_1_name}")
