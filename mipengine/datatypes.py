from enum import Enum


# class DType(Enum):
#     INT = 0
#     FLOAT = 1
#     STR = 2


# DTYPE2SQL = {
#     DType.INT: "INT",
#     DType.FLOAT: "FLOAT",
#     DType.STR: "TEXT",
# }
# SQL2DTYPE = mapping_inverse(DTYPE2SQL)

# DTYPE2PY = {
#     DType.INT: int,
#     DType.FLOAT: float,
#     DType.STR: str,
# }
# PY2DTYPE = mapping_inverse(DTYPE2PY)


class DType(Enum):
    """Members of DType represent data types in any language (python or sql).
    Each member has methods to_py and to_sql to convert its value to a concrete
    type.  There are also class methods from_py and from_sql to construct DType
    members from python/sql concrete types. The entire py2dtype and sql2dtype
    mappings are also provided ass class methods for convenience."""

    INT = int, "INT"
    FLOAT = float, "REAL"
    STR = str, "VARCHAR(50)"

    def __init__(self, pytype, sqltype):
        self._pytype = pytype
        self._sqltype = sqltype

    def to_py(self):
        return self._pytype

    def to_sql(self):
        return self._sqltype

    @classmethod
    def from_py(cls, pytype):
        mapping = cls.py2dtype()
        return mapping[pytype]

    @classmethod
    def from_sql(cls, sqltype):
        mapping = cls.sql2dtype()
        return mapping[sqltype]

    @classmethod
    def py2dtype(cls):
        mapping = {key: val for (key, _), val in cls._value2member_map_.items()}
        return mapping

    @classmethod
    def sql2dtype(cls):
        mapping = {key: val for (_, key), val in cls._value2member_map_.items()}
        # For many-to-one mappings add more items here
        # mapping.update({"VARCHAR(50)": mapping["TEXT"]})
        return mapping

    def __repr__(self):
        cls = type(self).__name__
        return f"{cls}.{self.name}"
