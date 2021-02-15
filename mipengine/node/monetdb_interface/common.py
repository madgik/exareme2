from typing import List
from typing import Union

import pymonetdb
from pymonetdb.sql.cursors import Cursor

# TODO What's the max text value we need?
from mipengine.node.config.config_parser import Config
from mipengine.node.tasks.data_classes import ColumnInfo
from mipengine.node.tasks.data_classes import TableInfo

MONETDB_VARCHAR_SIZE = 50

# TODO Add monetdb asyncio connection (aiopymonetdb)
config = Config().config
connection = pymonetdb.connect(username=config["monet_db"]["username"],
                               port=config["monet_db"]["port"],
                               password=config["monet_db"]["password"],
                               hostname=config["monet_db"]["hostname"],
                               database=config["monet_db"]["database"])
cursor: Cursor = connection.cursor()


def create_table_name(table_type: str, context_id: str, node_id: str) -> str:
    """
    Creates a table name with the format <table_type>_<context_id>_<node_id>_<uuid>
    """
    if table_type not in {"table", "view", "merge"}:
        raise KeyError(f"Table type is not acceptable: {table_type} .")
    if node_id not in {"global", config["node"]["identifier"]}:
        raise KeyError(f"Node Identifier is not acceptable: {node_id} .")

    uuid = str(pymonetdb.uuid.uuid1()).replace("-", "")

    return f"{table_type}_{context_id}_{node_id}_{uuid}"


# TODO Add SQLAlchemy if possible
# TODO We need to add the PRIVATE/OPEN table logic

def get_table_type_enumeration_value(table_type: str) -> int:
    """ Converts MIP Engine's table types to MonetDB's table types
     normal -> 0,
     view -> 1,
     merge -> 3,
     remote -> 5,
        """
    return {
        "normal": 0,
        "view": 1,
        "merge": 3,
        "remote": 5,
    }[table_type]


def convert_to_monetdb_column_type(column_type: str) -> str:
    """ Converts MIP Engine's int,float,text types to monetdb
    int -> integer
    float -> double
    text -> varchar(???)
    bool -> boolean
    clob -> clob
    """
    return {
        "int": "int",
        "float": "double",
        "text": f"varchar({MONETDB_VARCHAR_SIZE})",
        "bool": "bool",
        "clob": "clob",
    }[str.lower(column_type)]


def convert_from_monetdb_column_type(column_type: str) -> str:
    """ Converts MonetDB's types to MIP Engine's types
    int ->  int
    double  -> float
    varchar(???)  -> text
    boolean -> bool
    clob -> clob
    """
    return {
        "int": "int",
        "double": "float",
        "varchar": "text",
        "bool": "bool",
        "clob": "clob",
    }[str.lower(column_type)]

    if column_type not in type_mapping.keys():
        raise TypeError(f"Type {column_type} cannot be converted to monetdb column type.")

    return type_mapping.get(column_type)


def get_table_schema(table_type: str, table_name: str) -> List[ColumnInfo]:
    """Retrieves a schema for a specific table type and table name  from the monetdb.

        Parameters
        ----------
        table_type : str
            The type of the table
        table_name : str
            The name of the table

        Returns
        ------
        List[ColumnInfo]
            A schema which is a list of ColumnInfo's objects.
    """
    cursor.execute(f"SELECT columns.name, columns.type "
                   f"FROM columns "
                   f"RIGHT JOIN tables "
                   f"ON tables.id = columns.table_id "
                   f"WHERE "
                   f"tables.type = {str(get_table_type_enumeration_value(table_type))} "
                   f"AND "
                   f"tables.name = '{table_name}' "
                   f"AND "
                   f"tables.system=false;")

    return [ColumnInfo(table[0], convert_from_monetdb_column_type(table[1])) for table in cursor]


def get_tables_names(table_type: str, context_id: str) -> List[str]:
    """Retrieves a list of table names, which contain the context_id    from the monetdb.

        Parameters
        ----------
        table_type : str
            The type of the table
        context_id : str
            The id of the experiment

        Returns
        ------
        List[str]
            A list of table names.
    """
    type_clause = f"type = {str(get_table_type_enumeration_value(table_type))} AND"

    context_clause = f"name LIKE '%{context_id.lower()}%' AND"

    cursor.execute(
        "SELECT name FROM tables "
        "WHERE"
        f" {type_clause}"
        f" {context_clause} "
        "system = false")

    return [table[0] for table in cursor]


def convert_schema_to_sql_query_format(schema: List[ColumnInfo]):
    """Converts a table's schema to a sql query.

        Parameters
        ----------
        schema : List[ColumnInfo]
            The schema(list of ColumnInfo) of a table

        Returns
        ------
        str
            The schema in a sql query formatted string
    """
    columns_schema = ""
    for column_info in schema:
        columns_schema += f"{column_info.name} {convert_to_monetdb_column_type(column_info.type)}, "
    columns_schema = columns_schema[:-2]
    return columns_schema


def get_table_data(table_type: str, table_name: str) -> List[List[Union[str, int, float, bool]]]:
    """Retrieves the data of a table with specific type and name  from the monetdb.

        Parameters
        ----------
        table_type : str
            The type of the table
        table_name : str
            The name of the table

        Returns
        ------
        List[List[Union[str, int, float, bool]]
            The data of the table.
    """
    cursor.execute(
        f"SELECT {table_name}.* "
        f"FROM {table_name} "
        f"INNER JOIN tables ON tables.name = '{table_name}' "
        f"WHERE tables.system=false "
        f"AND tables.type = {str(get_table_type_enumeration_value(table_type))}")
    return cursor.fetchall()


def clean_up(context_id: str):
    """Deletes all tables of any type with name that contain a specific context_id from the monetdb.

        Parameters
        ----------
        context_id : str
            The id of the experiment
    """
    context_clause = f"name LIKE '%{context_id.lower()}%' AND"
    context_clause = ""
    cursor.execute(
        "SELECT name FROM tables "
        "WHERE"
        f" {context_clause} "
        "system = false")

    for table in cursor:
        if "view" in table[0]:
            cursor.execute(f"DROP VIEW if exists sys.{table[0]}")
        else:
            cursor.execute(f"DROP TABLE if exists {table[0]} cascade")

    connection.commit()
