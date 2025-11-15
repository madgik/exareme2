from __future__ import annotations

from pathlib import Path
from typing import List

import duckdb

from exareme2.worker import config as worker_config

_duckdb_path = Path(worker_config.duckdb.path)
_duckdb_path.parent.mkdir(parents=True, exist_ok=True)


def execute_and_fetchall(query: str) -> List:
    with duckdb.connect(str(_duckdb_path), read_only=False) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            return cursor.fetchall()
        finally:
            cursor.close()
