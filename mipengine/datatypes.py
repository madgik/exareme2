from enum import Enum

MONETDB_VARCHAR_SIZE = 500


class DType(Enum):
    """Members of DType represent data types in any language (python or sql).
    Each member has methods to_py and to_sql to convert its value to a concrete
    type.  There are also class methods from_py and from_sql to construct DType
    members from python/sql concrete types. The entire py2dtype and sql2dtype
    mappings are also provided as class methods for convenience."""

    INT = "INT"
    FLOAT = "FLOAT"
    STR = "STR"
    JSON = "JSON"
    BINARY = "BLOB"

    def __init__(self, sqltype):
        self._sqltype = sqltype

    def to_py(self):
        mapping = self.dtype2py()
        return mapping[self]

    def to_sql(self):
        mapping = self.dtype2sql()
        return mapping[self]

    @classmethod
    def from_py(cls, pytype):
        mapping = cls.py2dtype()
        return mapping[pytype]

    @classmethod
    def from_sql(cls, sql_type):
        mapping = cls.sql2dtype()
        return mapping[sql_type.upper()]
        # We convert to upper case the monet returns the types in lower-case

    # Creates a DType from a common data element sql type
    @classmethod
    def from_cde(cls, cde_type):
        mapping = {
            "int": cls.INT,
            "real": cls.FLOAT,
            "text": cls.STR,
        }
        return mapping[cde_type]

    @classmethod
    def dtype2py(cls):
        return {
            cls.INT: int,
            cls.FLOAT: float,
            cls.STR: str,
        }

    @classmethod
    def dtype2sql(cls):
        return {
            cls.INT: "INT",
            cls.FLOAT: "REAL",
            cls.STR: f"VARCHAR({MONETDB_VARCHAR_SIZE})",
            cls.JSON: "CLOB",  # (BUG) A monet udf return type cannot be of JSON type.
            cls.BINARY: "BLOB",
        }

    @classmethod
    def py2dtype(cls):
        return {val: key for key, val in cls.dtype2py().items()}

    @classmethod
    def sql2dtype(cls):
        mapping = {val: key for key, val in cls.dtype2sql().items()}
        mapping["DOUBLE"] = cls.FLOAT
        mapping["VARCHAR"] = cls.STR
        # MonetDB returns VARCHAR instead of VARCHAR(50) when
        return mapping

    def __repr__(self):
        cls = type(self).__name__
        return f"{cls}.{self.name}"
