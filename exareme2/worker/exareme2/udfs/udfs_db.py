from typing import List

from exareme2.worker.exareme2.monetdb.monetdb_facade import db_execute_query
from exareme2.worker.exareme2.monetdb.monetdb_facade import db_execute_udf


def run_udf(udf_defenitions: List[str], udf_exec_stmt):
    db_execute_query(";\n".join(udf_defenitions))
    db_execute_udf(udf_exec_stmt)
