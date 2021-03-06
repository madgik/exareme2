from mipengine.node.monetdb_interface.common_action import connection
from mipengine.node.monetdb_interface.common_action import cursor


def run_udf(udf_creation_stmt: str,
            udf_execution_query: str,
            ):
    cursor.execute(udf_creation_stmt)
    cursor.execute(udf_execution_query)
    connection.commit()
