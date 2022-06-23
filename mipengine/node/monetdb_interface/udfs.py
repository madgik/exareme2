from typing import List

from mipengine.node.monetdb_interface.monet_db_facade import execute


def run_udf(udf_statements: List[str]):
    for udf_stmt in udf_statements:
        execute(udf_stmt)
