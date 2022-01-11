from enum import Enum

VARCHAR_SIZE = 500


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
    BINARY = "BINARY"

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
        return mapping[sql_type.upper()]  # In SQL always print upper-case by convention

    @classmethod
    def from_cde(cls, cde_type):
        """Creates a DType from a common data element sql type."""
        mapping = {
            "int": cls.INT,
            "real": cls.FLOAT,
            "text": cls.STR,
        }
        return mapping[cde_type]

    @classmethod
    def dtype2py(cls):
        mapping = {val: key for key, val in cls.py2dtype().items()}
        # Add here items for many-to-one mapping
        mapping[cls.JSON] = str
        return mapping

    @classmethod
    def dtype2sql(cls):
        return {
            cls.INT: "INT",
            cls.FLOAT: "DOUBLE",
            cls.STR: f"VARCHAR({VARCHAR_SIZE})",
            cls.JSON: "CLOB",  # (BUG) A MonetDB udf cannot return a JSON type.
            cls.BINARY: "BLOB",
        }

    @classmethod
    def py2dtype(cls):
        return {
            int: cls.INT,
            float: cls.FLOAT,
            str: cls.STR,
            bytes: cls.BINARY,
        }

    @classmethod
    def sql2dtype(cls):
        mapping = {val: key for key, val in cls.dtype2sql().items()}
        # Add here items for many-to-one mapping
        mapping["VARCHAR"] = cls.STR  # MonetDB returns VARCHAR instead of VARCHAR(50)
        return mapping

    def __repr__(self):
        cls = type(self).__name__
        return f"{cls}.{self.name}"
