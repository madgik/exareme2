from enum import Enum


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


# ----- Methods related to type conversions (not yet unified) -----
# TODO Refactor into unified functions


def convert_mip_type_to_python_type(mip_type: str):
    """
    Converts MIP's types to the relative python class.

    The "MIP" type that this method is expecting is related to
    the "sql_type" enumerations contained in the CDEsMetadata.
    """
    type_mapping = {
        "int": int,
        "real": float,
        "text": str,
    }

    if mip_type not in type_mapping.keys():
        raise KeyError(
            f"MIP type '{mip_type}' cannot be converted to a python class type."
        )

    return type_mapping.get(mip_type)
