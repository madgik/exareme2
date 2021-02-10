from typing import List, Union

import pymonetdb
from pymonetdb.sql.cursors import Cursor

# TODO What's the max text value we need?
from mipengine.config.config_parser import Config
from mipengine.worker.tasks.data_classes import TableInfo, ColumnInfo

MONETDB_VARCHAR_SIZE = 50

# TODO Add monetdb asyncio connection (aiopymonetdb)
config = Config().config
connection = pymonetdb.connect(username=config["monet_db"]["username"],
                               port=config["monet_db"]["port"],
                               password=config["monet_db"]["password"],
                               hostname=config["monet_db"]["hostname"],
                               database=config["monet_db"]["database"])
cursor: Cursor = connection.cursor()


def tables_naming_convention(table_type: str, context_Id: str, node_id: str) -> str:
    if table_type not in {"table", "view", "merge"}:
        raise KeyError(f"Table type is not acceptable: {table_type} .")
    if node_id not in {"global", config["node"]["identifier"]}:
        raise KeyError(f"Node Identifier is not acceptable: {node_id} .")

    uuid = str(pymonetdb.uuid.uuid1()).replace("-", "")

    return f"{table_type}_{context_Id}_{node_id}_{uuid}"


# TODO Add SQLAlchemy if possible
# TODO We need to add the PRIVATE/OPEN table logic

def get_table_type_enumeration_value(table_type: str) -> int:
    # TODO add view value
    # TODO lower case first
    return {
        "normal": 0,
        "view": 1,
        "merge": 3,
        "remote": 5,
    }[table_type]


def convert_to_monetdb_column_type(column_type: str) -> str:
    # TODO Convert all to lower case
    """ Converts MIP Engine's INT,FLOAT,TEXT types to monetdb
    INT -> INTEGER
    FLOAT -> DOUBLE
    TEXT -> VARCHAR(???)
    BOOL -> BOOLEAN
    """
    return {
        "int": "INTEGER",
        "float": "DOUBLE",
        "text": f"VARCHAR({MONETDB_VARCHAR_SIZE})",
        "bool": "BOOLEAN",
    }[str.lower(column_type)]


def convert_from_monetdb_column_type(column_type: str) -> str:
    # TODO lower case
    """ Converts MonetDB's INTEGER,DOUBLE,VARCHAR,BOOLEAN types to MIP Engine's types
    INT ->  INT
    DOUBLE  -> FLOAT
    VARCHAR(???)  -> TEXT
    BOOLEAN -> BOOL
    """
    return {
        "int": "INT",
        "double": "FLOAT",
        "varchar": "TEXT",
        "bool": "BOOL",
    }[str.lower(column_type)]

    if column_type not in type_mapping.keys():
        raise TypeError(f"Type {column_type} cannot be converted to monetdb column type.")

    return type_mapping.get(column_type)


def get_table_schema(table_type: str, table_name: str) -> List[ColumnInfo]:
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
    type_clause = f"type = {str(get_table_type_enumeration_value(table_type))} AND"

    context_clause = f"name LIKE '%{context_id.lower()}%' AND"

    cursor.execute(
        "SELECT name FROM tables "
        "WHERE"
        f" {type_clause}"
        f" {context_clause} "
        "system = false")

    return [table[0] for table in cursor]


def convert_table_info_to_sql_query_format(table_info: TableInfo):
    columns_schema = ""
    for column_info in table_info.schema:
        columns_schema += f"{column_info.name} {convert_to_monetdb_column_type(column_info.type)}, "
    columns_schema = columns_schema[:-2]
    return columns_schema


def get_table_data(table_type: str, table_name: str) -> List[List[Union[str, int, float, bool]]]:
    cursor.execute(
        f"SELECT {table_name}.* "
        f"FROM {table_name} "
        f"INNER JOIN tables ON tables.name = '{table_name}' "
        f"WHERE tables.system=false "
        f"AND tables.type = {str(get_table_type_enumeration_value(table_type))}")
    return cursor.fetchall()


def clean_up(context_id: str = None):
    context_clause = ""
    if context_id is not None:
        context_clause = f"name LIKE '%{context_id.lower()}%' AND"

    cursor.execute(
        "SELECT name FROM tables "
        "WHERE"
        f" {context_clause} "
        "system = false")

    tables_names = [table[0] for table in cursor]
    for name in tables_names:
        cursor.execute(f"DROP TABLE if exists {name} cascade")
    connection.commit()
