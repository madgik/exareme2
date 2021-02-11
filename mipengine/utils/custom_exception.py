from typing import List

from mipengine.node.tasks.data_classes import TableInfo


class TableCannotBeFound(Exception):
    """Exception raised for errors while retrieving a table from a database.

    Attributes:
        tables -- tables which caused the error
        message -- explanation of the error
    """

    def __init__(self, tables: List[str]):
        self.tables = tables
        self.message = f"The following tables were not found : {tables}"
        super().__init__(self.message)


class IncompatibleSchemasMergeException(Exception):
    """Exception raised for errors while trying to merge tables with incompatible schemas.

    Attributes:
        table -- table which caused the error
        message -- explanation of the error
    """

    def __init__(self, table_infos: List[TableInfo]):
        self.table_infos = table_infos
        self.message = f"Tables to be added doesn't match MERGE TABLE schema : {table_infos}"
        super().__init__(self.message)
