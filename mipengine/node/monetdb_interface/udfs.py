import logging
from typing import List

from mipengine.node.monetdb_interface.monet_db_connection import MonetDB
from mipengine.node.monetdb_interface.monet_db_connection import monetdb


def run_udf(udf_statements: List[str], request_id):

    for udf_stmt in udf_statements:
        logging.info(f"{udf_stmt[0:20]=} { len(udf_statements)=} {request_id=}")
        monetdb.execute(udf_stmt)
        logging.info(f"{udf_stmt[0:20]=} {request_id=} DONE!")
