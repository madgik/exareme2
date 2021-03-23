from mipengine.node.monetdb_interface.common_action import connection
from mipengine.node.monetdb_interface.common_action import cursor


def run_udf(udf_creation_stmt: str,
            udf_execution_query: str,
            ):

    print(f"(udf.py::run_udf) udf_creation_stmt->\n{udf_creation_stmt}\nudf_execution_query-> \n{udf_execution_query}")
    if udf_creation_stmt:
    	cursor.execute(udf_creation_stmt)

    cursor.execute(udf_execution_query)
    connection.commit()
