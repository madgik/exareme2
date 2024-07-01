import sqlite3
from typing import List

from exareme2.worker import config as worker_config

CONN = sqlite3.connect(
    f"{str(worker_config.data_path)}/{worker_config.sqlite.db_name}.db"
)


def execute_and_fetchall(query) -> List:
    cur = CONN.cursor()
    cur.execute(query)
    result = cur.fetchall()
    cur.close()
    return result
