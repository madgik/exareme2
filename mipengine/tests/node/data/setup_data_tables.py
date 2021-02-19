import os

import pymonetdb
from pymonetdb import Connection

from mipengine.common.node_catalog import LocalNode
from mipengine.common.node_catalog import NodeCatalog
from mipengine.node.config.config_parser import Config

config = Config().config
node_catalog = NodeCatalog()


def setup_data_table(connection: Connection, cursor, node_id: str):
    script_dir = os.path.dirname(__file__)
    data_table_creation_path = os.path.join(script_dir, "create_data_table.sql")
    columns = open(data_table_creation_path, "r")
    cursor.execute("DROP TABLE IF EXISTS data CASCADE")
    cursor.execute(columns.read())
    data_path = os.path.join(script_dir, f"{node_id}.sql")
    data = open(data_path, "r")
    cursor.execute(data.read())
    connection.commit()


def setup_database_for_node(local_node: LocalNode):
    return pymonetdb.connect(username=config["monet_db"]["username"],
                             port=local_node.monetdbPort,
                             password=config["monet_db"]["password"],
                             hostname=local_node.monetdbHostname,
                             database=config["monet_db"]["database"])


local_nodes = node_catalog.get_local_nodes()
for local_node in local_nodes:
    current_connection = setup_database_for_node(local_node)
    current_cursor = current_connection.cursor()
    setup_data_table(current_connection, current_cursor, local_node.nodeId)
