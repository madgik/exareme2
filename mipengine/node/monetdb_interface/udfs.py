from mipengine.node.monetdb_interface.monet_db_connection import MonetDB


def run_udf(udf_creation_stmt: str,
            udf_execution_query: str,
            ):
    if udf_creation_stmt:
        MonetDB().execute(udf_creation_stmt)
    MonetDB().execute(udf_execution_query)
