"""
!!!!!!!!!!!!!!!!!!!!!! ATTENTION !!!!!!!!!!!!!!!!!!!!!!!!
In some cases an exception thrown by the NODE(celery) will be received
in the CONTROLLER(celery get method) as a generic Exception and catching
it by it's definition won't be possible.

This is happening due to a celery problem: https://github.com/celery/celery/issues/3586

There are some workarounds possible that are case specific.

For example in the DataModelUnavailable using the `super().__init__(self.message)`
was creating many problems in deserializing the exception.

When adding a new exception, the task throwing it should be tested:
1) That you can catch the exception by it's name,
2) the contained message, if exists, is shown properly.
"""
from typing import List


class TablesNotFound(Exception):
    """
    Exception raised for errors while retrieving a table from a database.

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

    def __init__(self, table_names: List[str]):
        self.table_names = table_names
        self.message = (
            f"Tables to be added don't match MERGE TABLE schema : {table_names}"
        )
        super().__init__(self.message)


class IncompatibleTableTypes(Exception):
    """Exception raised for errors while trying to merge tables with incompatible table types.

    Attributes:
        table_types --  the types of the table which caused the error
        message -- explanation of the error
    """

    def __init__(self, table_types: set):
        self.table_types = table_types
        self.message = f"Tables have more than one distinct types : {self.table_types}"
        super().__init__(self.message)


class RequestIDNotFound(Exception):
    """Exception raised while checking the presence of request_id in task's arguments.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self):
        self.message = f"Request id is missing from task's arguments."
        super().__init__(self.message)


class DataModelUnavailable(Exception):
    """
    Exception raised when a data model is not available in the NODE db.

    Attributes:
        node_id -- the node id that threw the exception
        data_model --  the unavailable data model
        message -- explanation of the error
    """

    def __init__(self, node_id: str, data_model: str):
        self.node_id = node_id
        self.data_model = data_model
        self.message = f"Data model '{self.data_model}' is not available in node: '{self.node_id}'."


class DatasetUnavailable(Exception):
    """
    Exception raised when a dataset is not available in the NODE db.

    Attributes:
        node_id -- the node id that threw the exception
        dataset --  the unavailable dataset
        message -- explanation of the error
    """

    def __init__(self, node_id: str, dataset: str):
        self.node_id = node_id
        self.dataset = dataset
        self.message = (
            f"Dataset '{self.dataset}' is not available in node: '{self.node_id}'."
        )


class InsufficientDataError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class BadRequest(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class BadUserInput(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
