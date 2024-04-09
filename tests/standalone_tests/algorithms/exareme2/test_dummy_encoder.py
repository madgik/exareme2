from types import SimpleNamespace

from exareme2.algorithms.exareme2.preprocessing import DummyEncoderUdf


def test_dummy_encoder__only_numerical_vars():
    enums = {}
    numerical_vars = ["n1", "n2"]
    design_matrix = SimpleNamespace()
    design_matrix.name = "test_table"
    kwargs = {
        "x": design_matrix,
        "enums": enums,
        "numerical_vars": numerical_vars,
        "intercept": True,
    }

    expected = """\
INSERT INTO __main
SELECT
    "row_id",
    1 AS "intercept",
    "n1",
    "n2"
FROM
    test_table;"""

    enc = DummyEncoderUdf(flowkwargs=kwargs)
    result = enc.get_exec_stmt(udf_name=None, output_table_names=["__main"])
    assert result == expected


def test_dummy_encoder__only_nominal_vars():
    enums = {
        "c1": [{"code": "l1", "dummy": "c1__1"}, {"code": "l2", "dummy": "c1__2"}],
    }
    numerical_vars = []
    design_matrix = SimpleNamespace()
    design_matrix.name = "test_table"
    kwargs = {
        "x": design_matrix,
        "enums": enums,
        "numerical_vars": numerical_vars,
        "intercept": True,
    }

    expected = """\
INSERT INTO __main
SELECT
    "row_id",
    1 AS "intercept",
    CASE WHEN c1 = 'l1' THEN 1 ELSE 0 END AS "c1__1",
    CASE WHEN c1 = 'l2' THEN 1 ELSE 0 END AS "c1__2"
FROM
    test_table;"""

    enc = DummyEncoderUdf(flowkwargs=kwargs)
    result = enc.get_exec_stmt(udf_name=None, output_table_names=["__main"])
    assert result == expected


def test_dummy_encoder__no_intercept():
    enums = {
        "c1": [{"code": "l1", "dummy": "c1__1"}, {"code": "l2", "dummy": "c1__2"}],
    }
    numerical_vars = []
    design_matrix = SimpleNamespace()
    design_matrix.name = "test_table"
    kwargs = {
        "x": design_matrix,
        "enums": enums,
        "numerical_vars": numerical_vars,
        "intercept": False,
    }

    expected = """\
INSERT INTO __main
SELECT
    "row_id",
    CASE WHEN c1 = 'l1' THEN 1 ELSE 0 END AS "c1__1",
    CASE WHEN c1 = 'l2' THEN 1 ELSE 0 END AS "c1__2"
FROM
    test_table;"""

    enc = DummyEncoderUdf(flowkwargs=kwargs)
    result = enc.get_exec_stmt(udf_name=None, output_table_names=["__main"])
    assert result == expected


def test_dummy_encoder__mixed_vars():
    enums = {
        "c1": [{"code": "l1", "dummy": "c1__1"}, {"code": "l2", "dummy": "c1__2"}],
        "c2": [
            {"code": "A", "dummy": "c2__1"},
            {"code": "B", "dummy": "c2__2"},
            {"code": "C", "dummy": "c2__3"},
        ],
    }
    numerical_vars = ["n1", "n2"]
    design_matrix = SimpleNamespace()
    design_matrix.name = "test_table"
    kwargs = {
        "x": design_matrix,
        "enums": enums,
        "numerical_vars": numerical_vars,
        "intercept": True,
    }

    expected = """\
INSERT INTO __main
SELECT
    "row_id",
    1 AS "intercept",
    CASE WHEN c1 = 'l1' THEN 1 ELSE 0 END AS "c1__1",
    CASE WHEN c1 = 'l2' THEN 1 ELSE 0 END AS "c1__2",
    CASE WHEN c2 = 'A' THEN 1 ELSE 0 END AS "c2__1",
    CASE WHEN c2 = 'B' THEN 1 ELSE 0 END AS "c2__2",
    CASE WHEN c2 = 'C' THEN 1 ELSE 0 END AS "c2__3",
    "n1",
    "n2"
FROM
    test_table;"""

    enc = DummyEncoderUdf(flowkwargs=kwargs)
    result = enc.get_exec_stmt(udf_name=None, output_table_names=["__main"])
    assert result == expected


def test_dummy_encoder__result_create_query():
    enums = {
        "c1": [{"code": "l1", "dummy": "c1__1"}, {"code": "l2", "dummy": "c1__2"}],
        "c2": [
            {"code": "A", "dummy": "c2__1"},
            {"code": "B", "dummy": "c2__2"},
            {"code": "C", "dummy": "c2__3"},
        ],
    }
    numerical_vars = ["n1", "n2"]
    design_matrix = SimpleNamespace()
    design_matrix.name = "test_table"
    kwargs = {
        "x": design_matrix,
        "enums": enums,
        "numerical_vars": numerical_vars,
        "intercept": True,
    }

    expected = 'CREATE TABLE __main("row_id" INT,"intercept" DOUBLE,"c1__1" DOUBLE,"c1__2" DOUBLE,"c2__1" DOUBLE,"c2__2" DOUBLE,"c2__3" DOUBLE,"n1" DOUBLE,"n2" DOUBLE);'

    enc = DummyEncoderUdf(flowkwargs=kwargs)
    result = enc.get_results(output_table_names=["__main"])

    assert result[0].create_query == expected
