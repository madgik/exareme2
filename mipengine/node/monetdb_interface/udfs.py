from typing import List

from mipengine.node.monetdb_interface.monet_db_connection import MonetDBPool


def run_udf(udf_statements: List[str]):
    for udf_stmt in udf_statements:
        MonetDBPool().execute(udf_stmt)
