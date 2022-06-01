import logging
from typing import List

from mipengine.node.monetdb_interface.monet_db_connection import DBExecutionDTO
from mipengine.node.monetdb_interface.monet_db_connection import MonetDB


def run_udf(udf_statements: List[str]):

    for udf_stmt in udf_statements:
        MonetDB().execute(DBExecutionDTO(udf_stmt))
