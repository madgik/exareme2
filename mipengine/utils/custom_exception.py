class TableCannotBeFound(Exception):
    """Exception raised for errors while retrieving a table from a database.

    Attributes:
        table -- table which caused the error
        message -- explanation of the error
    """

    def __init__(self, table):
        self.table = table
        self.message = f"no such table {table} in schema 'sys'"
        super().__init__(self.message)


class IncompatibleSchemasMergeException(Exception):
    """Exception raised for errors while trying to merge tables with incompatible schemas.

    Attributes:
        table -- table which caused the error
        message -- explanation of the error
    """

    def __init__(self, table):
        self.table = table
        self.message = f"{table} to be added doesn't match MERGE TABLE schema"
        super().__init__(self.message)
