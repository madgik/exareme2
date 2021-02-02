from typing import List, Union

import pymonetdb
from pymonetdb.sql.cursors import Cursor

from worker.config.config_parser import Config
# TODO What's the max text value we need?
from worker.tasks.data_classes import ColumnInfo, TableInfo

MONETDB_VARCHAR_SIZE = 50

# TODO Add monetdb asyncio connection (aiopymonetdb)
# TODO Read these from config
config = Config().config
connection = pymonetdb.connect(username=config["monet_db"]["username"], password=config["monet_db"]["password"],
                               hostname=config["monet_db"]["hostname"], database=config["monet_db"]["database"])
cursor: Cursor = connection.cursor()


# TODO Add SQLAlchemy if possible
# TODO We need to add the PRIVATE/OPEN table logic

def get_table_type_enumeration_value(table_type: str) -> int:
    # TODO add view value
    # TODO lower case first
    return {
        "normal": 0,
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


def get_table_schema(table_name: str = None) -> List[ColumnInfo]:
    return get_tables_info([table_name])[0].schema


def get_tables_info(table_names: List[str] = None) -> List[TableInfo]:
    return __get_tables_info("normal", table_names)


def __get_tables_info(table_type: str = None, table_names: List[str] = None) -> List[TableInfo]:
    type_clause = ''
    if table_type is not None:
        type_clause = "tables.type = " + str(get_table_type_enumeration_value(table_type)) + " AND "

    name_clause = ''
    if table_names is not None:
        for table_name in table_names:
            name_clause += f"tables.name = '{table_name}' AND "

    cursor.execute(
        "SELECT  tables.name, columns.type, columns.name FROM tables "
        "RIGHT JOIN columns ON tables.id = columns.table_id "
        "WHERE " + type_clause + name_clause +
        "tables.system=false")

    table_infos = convert_tables_query_cursor_to_tables_info(cursor)
    if table_names is not None and len(table_names) != len(table_infos):
        raise KeyError(f"Could not find all table names: {table_names} .")
    return table_infos


def convert_tables_query_cursor_to_tables_info(tables_query_cursor: Cursor) -> List[TableInfo]:
    tables = {}
    for (table_name, column_type, column_name) in tables_query_cursor:

        table_column = ColumnInfo(column_name, convert_from_monetdb_column_type(column_type))
        if table_name not in tables:
            tables[table_name] = [table_column]
        else:
            tables[table_name].append(table_column)

    return [TableInfo(table_name, table_columns) for table_name, table_columns in tables.items()]


def create_table(table_info: TableInfo):
    columns_schema = "("
    for column_info in table_info.schema:
        columns_schema += f"{column_info['name']} {convert_to_monetdb_column_type(column_info['type'])}, "
    columns_schema = columns_schema[:-2]

    cursor.execute(f"CREATE TABLE {table_info.name} {columns_schema} )")
    connection.commit()


def get_table_data(table_name: str) -> List[List[Union[str, int, float, bool]]]:
    cursor.execute(f"SELECT * FROM {table_name} ")
    return cursor.fetchall()


def delete_table(table_name: str):
    __delete_table(table_name, "normal")


def delete_merge_table(table_name: str):
    __delete_table(table_name, "merge")


def delete_remote_table(table_name: str):
    __delete_table(table_name, "remote")


def __delete_table(table_name: str, table_type: str):
    print(f"Delete call{table_name}")

    cursor.execute(f"SELECT type FROM tables where name = '{table_name}'")
    table_type_in_database = cursor.fetchall()[0][0]
    if table_type_in_database != get_table_type_enumeration_value(table_type):
        raise TypeError(f"Table {table_name} is not of type {table_type}")

    print(f"DROP TABLE {table_name}")
    print(f"TYPE {table_type_in_database}")
    cursor.execute(f"DROP TABLE {table_name}")
    connection.commit()
