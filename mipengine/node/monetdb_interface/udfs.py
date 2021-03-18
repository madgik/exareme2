from mipengine.node.monetdb_interface.monet_db_connection import get_connection


def run_udf(udf_creation_stmt: str,
            udf_execution_query: str,
            ):

    connection = get_connection()
    cursor = connection.cursor()
    if udf_creation_stmt:
        cursor.execute(udf_creation_stmt)
    cursor.execute(udf_execution_query)
