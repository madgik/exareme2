from enum import Enum


class DType(Enum):
    """Members of DType represent data types in any language (python or sql).
    Each member has methods to_py and to_sql to convert its value to a concrete
    type.  There are also class methods from_py and from_sql to construct DType
    members from python/sql concrete types. The entire py2dtype and sql2dtype
    mappings are also provided ass class methods for convenience."""

    INT = "INT"
    FLOAT = "REAL"
    STR = "VARCHAR(50)"

    def __init__(self, sqltype):
        self._sqltype = sqltype

    def to_py(self):
        return self.dtype2pytype_mapping().get(self.name)

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
    def from_monet_type(cls, monet_type):
        mapping = cls.monet2dtype_mapping()
        return mapping[monet_type]

    @classmethod
    def monet2dtype_mapping(cls):
        return {
            "int": cls.INT,
            "double": cls.FLOAT,
            "real": cls.FLOAT,
            "varchar": cls.STR,
        }

    @classmethod
    def dtype2pytype_mapping(cls):
        return {
            cls.INT: int,
            cls.FLOAT: float,
            cls.STR: str,
        }

    @classmethod
    def py2dtype(cls):
        return {val: key for key, val in cls.dtype2pytype_mapping().items()}

    @classmethod
    def sql2dtype(cls):
        mapping = {key: val for (_, key), val in cls._value2member_map_.items()}
        # For many-to-one mappings add more items here
        # mapping.update({"VARCHAR(50)": mapping["TEXT"]})
        return mapping

    def __repr__(self):
        cls = type(self).__name__
        return f"{cls}.{self.name}"


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
