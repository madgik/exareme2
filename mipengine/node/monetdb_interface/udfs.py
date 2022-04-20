import logging
from typing import List

from billiard.process import current_process

from mipengine.node.monetdb_interface.monet_db_connection import MonetDB


def run_udf(udf_statements: List[str]):
    for udf_stmt in udf_statements:
        logging.error(f"{current_process().index=}")
        logging.error(f"{udf_stmt=}")
        MonetDB().execute(udf_stmt)
