import json
import re
from contextlib import contextmanager
from sys import stdin

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
@click.option("--port", default=50000, type=int, help="The port of the database.")
def show_node_db_actions(ip, port):
    with db_cursor(ip, port) as cur:
        cur.execute(f"select * from {DB_METADATA_SCHEMA}.{ACTIONS_TABLE};")
        results = cur.fetchall()
        for _, action_str in results:
            action = json.loads(action_str)
            if (
                action["action"] == ADD_DATA_MODEL_ACTION_CODE
                or action["action"] == DELETE_DATA_MODEL_ACTION_CODE
            ):
                print(
                    f"{action['date']} - {action['user']} - {action['action']} - {action['data_model_code']}:{action['data_model_version']} - {action['data_model_label']}"
                )
            elif (
                action["action"] == ADD_DATASET_ACTION_CODE
                or action["action"] == DELETE_DATASET_ACTION_CODE
            ):
                print(
                    f"{action['date']} - {action['user']} - {action['action']} - {action['dataset_code']} - {action['dataset_label']} - {action['data_model_code']}:{action['data_model_version']} - {action['data_model_label']}"
                )


LOG_FILE_CHUNK_SIZE = 1024  # Will read the logfile in chunks
TIMESTAMP_REGEX = (
    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}"  # 2022-04-13 18:25:22,875
)
NODE_JOINED_PATTERN = (
    rf"({TIMESTAMP_REGEX}) .* Node with id '(.*)' joined the federation.$"
)
NODE_LEFT_PATTERN = rf"({TIMESTAMP_REGEX}) .* Node with id '(.*)' left the federation.$"
DATA_MODEL_ADDED_PATTERN = rf"({TIMESTAMP_REGEX}) .* Datamodel '(.*)' was added.$"
DATA_MODEL_REMOVED_PATTERN = rf"({TIMESTAMP_REGEX}) .* Datamodel '(.*)' was removed.$"

DATASET_ADDED_PATTERN = rf"({TIMESTAMP_REGEX}) .* Dataset '(.*)' of datamodel '(.*)' was \
added in node '(.*)'.$"

DATASET_REMOVED_PATTERN = rf"({TIMESTAMP_REGEX}) .* Dataset '(.*)' of datamodel '(.*)' \
was removed from node '(.*)'.$"

EXPERIMENT_EXECUTION_PATTERN = rf"({TIMESTAMP_REGEX}) .* Experiment with request id \
'(.*)' and context id '(.*)' is starting algorithm '(.*)', touching datasets '(.*)' on \
local nodes '(.*)' with parameters '(.*)'.$"


def print_audit_entry(log_line):
    if pattern_groups := re.search(NODE_JOINED_PATTERN, log_line):
        print(f"{pattern_groups.group(1)} - NODE_JOINED - {pattern_groups.group(2)}")
    elif pattern_groups := re.search(NODE_LEFT_PATTERN, log_line):
        print(f"{pattern_groups.group(1)} - NODE_LEFT - {pattern_groups.group(2)}")
    elif pattern_groups := re.search(DATA_MODEL_ADDED_PATTERN, log_line):
        print(
            f"{pattern_groups.group(1)} - DATAMODEL_ADDED - {pattern_groups.group(2)}"
        )
    elif pattern_groups := re.search(DATA_MODEL_REMOVED_PATTERN, log_line):
        print(
            f"{pattern_groups.group(1)} - DATAMODEL_REMOVED - {pattern_groups.group(2)}"
        )
    elif pattern_groups := re.search(DATASET_ADDED_PATTERN, log_line):
        print(
            f"{pattern_groups.group(1)} - DATASET_ADDED - {pattern_groups.group(4)} - {pattern_groups.group(3)} - {pattern_groups.group(2)}"
        )
    elif pattern_groups := re.search(DATASET_REMOVED_PATTERN, log_line):
        print(
            f"{pattern_groups.group(1)} - DATASET_REMOVED - {pattern_groups.group(4)} - {pattern_groups.group(3)} - {pattern_groups.group(2)}"
        )
    elif pattern_groups := re.search(EXPERIMENT_EXECUTION_PATTERN, log_line):
        print(
            f"{pattern_groups.group(1)} - EXPERIMENT_STARTED - {pattern_groups.group(2)} - {pattern_groups.group(4)} - {pattern_groups.group(5)} - {pattern_groups.group(7)}"
        )


@cli.command()
@click.option(
    "--logfile",
    help="The logfile to get the audit entries from. Will use stdin if not provided.",
    type=click.File("r"),
    default=stdin,
)
def show_controller_audit_entries(logfile):

    previous_chunk_remains = ""
    while logs_chunk := logfile.read(LOG_FILE_CHUNK_SIZE):
        logs_chunk = previous_chunk_remains + logs_chunk
        # Separate lines when "\n2022-04-13 18:25:22,875 - is found
        separate_log_lines = re.split(rf"\n(?={TIMESTAMP_REGEX} -)", logs_chunk)

        # The final log_line could be incomplete due to "chunking"
        for log_line in separate_log_lines[:-1]:
            print_audit_entry(log_line)
        previous_chunk_remains = separate_log_lines[-1]
    else:
        print_audit_entry(previous_chunk_remains)


if __name__ == "__main__":
    cli()
