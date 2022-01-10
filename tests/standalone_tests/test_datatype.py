from mipengine import DType


def test_py_conversion():
    for tp in [DType.INT, DType.FLOAT, DType.STR, DType.BINARY]:
        assert DType.from_py(tp.to_py()) == tp

    # JSON special case due to MonetDB UDFs not being able to return JSON type
    tp = DType.JSON
    assert DType.from_py(tp.to_py()) == DType.STR


def test_py_conversion_inverse():
    for tp in [int, float, str, bytes]:
        assert DType.py2dtype()[tp].to_py() == tp


def test_sql_conversion():
    for tp in DType:
        assert DType.from_sql(tp.to_sql()) == tp


def test_sql_conversion_inverse():
    for tp in ["INT", "DOUBLE", "VARCHAR(500)", "CLOB", "BLOB"]:
        assert DType.sql2dtype()[tp].to_sql() == tp

    # VARCHAR special case
    tp = "VARCHAR"
    assert DType.sql2dtype()[tp].to_sql() == "VARCHAR(500)"
