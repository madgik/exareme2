import json
from contextlib import contextmanager

import click
import pymonetdb

DB_USERNAME = "monetdb"
DB_PASSWORD = "monetdb"
DB_FARM = "db"
DB_METADATA_SCHEMA = "mipdb_metadata"
ACTIONS_TABLE = "actions"
ADD_DATA_MODEL_ACTION_CODE = "ADD DATA MODEL"
DELETE_DATA_MODEL_ACTION_CODE = "DELETE DATA MODEL"
ADD_DATASET_ACTION_CODE = "ADD DATASET"
DELETE_DATASET_ACTION_CODE = "DELETE DATASET"


@contextmanager
def db_cursor(ip, port):
    connection = pymonetdb.connect(
        hostname=ip,
        port=port,
        username=DB_USERNAME,
        password=DB_PASSWORD,
        database=DB_FARM,
        autocommit=True,
    )
    cursor = connection.cursor()
    yield cursor
    cursor.close()
    connection.close()


@click.group()
def cli():
    """
    This is a log aggregation script.
    It can be used either in a local hospital node to show database actions or in the federation master node
    to show information for all the federation nodes.
    """
    pass


@cli.command()
@click.option("--ip", default="127.0.0.1", help="The ip of the database.")
@click.option("--port", required=True, type=int, help="The port of the database.")
def show_node_db_actions(ip, port):
    with db_cursor(ip, port) as cur:
        cur.execute(f"select * from {DB_METADATA_SCHEMA}.{ACTIONS_TABLE};")
        results = cur.fetchall()
        for _, action_str in results:
            action = json.loads(action_str)
            if action["action"] == ADD_DATA_MODEL_ACTION_CODE:
                print(
                    f"Data model '{action['data_model_label']}' with code '{action['data_model_code']}:{action['data_model_version']}' ADDED from user '{action['user']}' at '{action['date']}'."
                )
            elif action["action"] == DELETE_DATA_MODEL_ACTION_CODE:
                print(
                    f"Data model '{action['data_model_label']}' with code '{action['data_model_code']}:{action['data_model_version']}'  DELETED from user '{action['user']}' at '{action['date']}'."
                )
            elif action["action"] == ADD_DATASET_ACTION_CODE:
                print(
                    f"Dataset '{action['dataset_label']}' with code '{action['dataset_code']}' of data model '{action['data_model_code']}:{action['data_model_version']}' ADDED from user '{action['user']}' at '{action['date']}'."
                )
            elif action["action"] == DELETE_DATASET_ACTION_CODE:
                print(
                    f"Dataset '{action['dataset_label']}' with code '{action['dataset_code']}' of data model '{action['data_model_code']}:{action['data_model_version']}' DELETED from user '{action['user']}' at '{action['date']}'."
                )


if __name__ == "__main__":
    cli()
