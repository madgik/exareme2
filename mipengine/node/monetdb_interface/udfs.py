from typing import List

from mipengine.node.monetdb_interface.monet_db_facade import db_execute_query
from mipengine.node.monetdb_interface.monet_db_facade import db_execute_udf


def run_udf(udf_defenitions: List[str], udf_exec_stmt):
    for udf_stmt in udf_defenitions:
        db_execute_query(udf_stmt)
    db_execute_udf(udf_exec_stmt)
