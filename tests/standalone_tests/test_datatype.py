from exareme2 import DType


def test_py_conversion():
    for dtype in [DType.INT, DType.FLOAT, DType.STR, DType.BINARY]:
        assert DType.from_py(dtype.to_py()) == dtype

    # JSON special case due to MonetDB UDFs not being able to return JSON type
    dtype = DType.JSON
    assert DType.from_py(dtype.to_py()) == DType.STR


def test_py_conversion_inverse():
    for dtype in [int, float, str, bytes]:
        assert DType.py2dtype()[dtype].to_py() == dtype


def test_sql_conversion():
    for dtype in DType:
        assert DType.from_sql(dtype.to_sql()) == dtype


def test_sql_conversion_inverse():
    for dtype in ["INT", "DOUBLE", "VARCHAR(500)", "CLOB", "BLOB"]:
        assert DType.sql2dtype()[dtype].to_sql() == dtype

    # VARCHAR special case
    dtype = "VARCHAR"
    assert DType.sql2dtype()[dtype].to_sql() == "VARCHAR(500)"
