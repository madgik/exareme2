from typing import List

from exareme2.worker.exareme2.monetdb import monetdb_facade


def run_udf(udf_defenitions: List[str], udf_exec_stmt):
    monetdb_facade.execute_query(";\n".join(udf_defenitions))
    monetdb_facade.execute_udf(udf_exec_stmt)
