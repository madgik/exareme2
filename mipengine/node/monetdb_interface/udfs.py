from mipengine.node.monetdb_interface.monet_db_connection import execute_with_occ


def run_udf(udf_creation_stmt: str,
            udf_execution_query: str,
            ):
    if udf_creation_stmt:
        execute_with_occ(udf_creation_stmt)
    execute_with_occ(udf_execution_query)
