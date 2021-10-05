from enum import Enum


class DType(Enum):
    """Members of DType represent data types in any language (python or sql).
    Each member has methods to_py and to_sql to convert its value to a concrete
    type.  There are also class methods from_py and from_sql to construct DType
    members from python/sql concrete types. The entire py2dtype and sql2dtype
    mappings are also provided ass class methods for convenience."""

    INT = "INT"
    FLOAT = "FLOAT"
    STR = "STR"

    def __init__(self, sqltype):
        self._sqltype = sqltype

    def to_py(self):
        mapping = self.dtype2py()
        return mapping[self]

    def to_sql(self):
        mapping = {
            self.INT: "INT",
            self.FLOAT: "REAL",
            self.STR: "VARCHAR(50)",
        }
        return mapping[self]

    @classmethod
    def from_py(cls, pytype):
        mapping = cls.py2dtype()
        return mapping[pytype]

    @classmethod
    def from_sql(cls, sql_type):
        mapping = {
            "int": cls.INT,
            "double": cls.FLOAT,
            "real": cls.FLOAT,
            "varchar": cls.STR,
        }
        return mapping[sql_type]

    @classmethod
    def from_metadata(cls, metadata_type):
        mapping = {
            "int": cls.INT,
            "real": cls.FLOAT,
            "text": cls.STR,
        }
        return mapping[metadata_type]

    @classmethod
    def dtype2py(cls):
        return {
            cls.INT: int,
            cls.FLOAT: float,
            cls.STR: str,
        }

    @classmethod
    def py2dtype(cls):
        return {val: key for key, val in cls.dtype2py().items()}

    def __repr__(self):
        cls = type(self).__name__
        return f"{cls}.{self.name}"
