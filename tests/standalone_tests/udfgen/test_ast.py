# type: ignore
from mipengine.udfgen.ast import Column
from mipengine.udfgen.ast import ScalarFunction
from mipengine.udfgen.ast import Select
from mipengine.udfgen.ast import Table
from mipengine.udfgen.ast import TableFunction


def test_column_alias():
    col = Column("very_long_name", alias="name")
    result = col.compile()
    expected = '"very_long_name" AS name'
    assert result == expected


def test_column_mul():
    col1 = Column("col1", alias="column1")
    col2 = Column("col2", alias="column2")
    result = (col1 * col2).compile()
    expected = '"col1" * "col2"'
    assert result == expected


def test_column_add():
    col1 = Column("col1", alias="column1")
    col2 = Column("col2", alias="column2")
    result = (col1 + col2).compile()
    expected = '"col1" + "col2"'
    assert result == expected


def test_column_sub():
    col1 = Column("col1", alias="column1")
    col2 = Column("col2", alias="column2")
    result = (col1 - col2).compile()
    expected = '"col1" - "col2"'
    assert result == expected


def test_column_div():
    col1 = Column("col1", alias="column1")
    col2 = Column("col2", alias="column2")
    result = (col1 / col2).compile()
    expected = '"col1" / "col2"'
    assert result == expected


def test_column_mul_from_table():
    tab = Table(name="tab", columns=["a", "b"])
    prod = tab.c["a"] * tab.c["b"]
    result = prod.compile()
    expected = 'tab."a" * tab."b"'
    assert result == expected


def test_select_single_table():
    tab = Table(name="a_table", columns=["c1", "c2"])
    sel = Select([tab.c["c1"], tab.c["c2"]], [tab])
    result = sel.compile()
    expected = """\
SELECT
    a_table."c1",
    a_table."c2"
FROM
    a_table"""
    assert result == expected


def test_select_two_tables_joined():
    tab1 = Table(name="tab1", columns=["a", "b"])
    tab2 = Table(name="tab2", columns=["c", "d"])
    sel = Select([tab1.c["a"], tab1.c["b"], tab2.c["c"], tab2.c["d"]], [tab1, tab2])
    expected = """\
SELECT
    tab1."a",
    tab1."b",
    tab2."c",
    tab2."d"
FROM
    tab1,
    tab2"""
    result = sel.compile()
    assert result == expected


def test_select_join_on_one_column():
    tab1 = Table(name="tab1", columns=["a", "b"])
    tab2 = Table(name="tab2", columns=["c", "b"])
    sel = Select(
        [tab1.c["a"], tab1.c["b"], tab2.c["c"]],
        [tab1, tab2],
        where=[tab1.c["b"] == tab2.c["b"]],
    )
    result = sel.compile()
    expected = '''\
SELECT
    tab1."a",
    tab1."b",
    tab2."c"
FROM
    tab1,
    tab2
WHERE
    tab1."b"=tab2."b"'''
    assert result == expected


def test_select_join_on_two_columns():
    tab1 = Table(name="tab1", columns=["a", "b", "c"])
    tab2 = Table(name="tab2", columns=["c", "b", "d"])
    sel = Select(
        [tab1.c["a"], tab1.c["b"], tab1.c["c"], tab2.c["d"]],
        [tab1, tab2],
        where=[tab1.c["b"] == tab2.c["b"], tab1.c["c"] == tab2.c["c"]],
    )
    result = sel.compile()
    expected = '''\
SELECT
    tab1."a",
    tab1."b",
    tab1."c",
    tab2."d"
FROM
    tab1,
    tab2
WHERE
    tab1."b"=tab2."b" AND
    tab1."c"=tab2."c"'''
    assert result == expected


def test_select_star():
    tab1 = Table(name="tab1", columns=["a", "b", "c"])
    sel = Select([Column("*")], [tab1])
    result = sel.compile()
    expected = """\
SELECT
    *
FROM
    tab1"""
    assert result == expected


def test_select_scalar_returning_func():
    tab1 = Table(name="tab1", columns=["a", "b"])
    func = ScalarFunction(name="the_func", columns=[tab1.c["a"], tab1.c["b"]])
    sel = Select([func], [tab1])
    result = sel.compile()
    expected = """\
SELECT
    the_func(tab1."a",tab1."b")
FROM
    tab1"""
    assert result == expected


def test_select_table_returning_func():
    tab = Table(name="tab", columns=["a", "b"])
    func = TableFunction(
        name="the_func", subquery=Select([tab.c["a"], tab.c["b"]], [tab])
    )
    sel = Select([Column("*")], [func])
    result = sel.compile()
    expected = """\
SELECT
    *
FROM
    the_func((
        SELECT
            tab."a",
            tab."b"
        FROM
            tab
    ))"""
    assert result == expected


def test_select_star_and_column_from_table_returning_func():
    tab = Table(name="tab", columns=["a", "b"])
    func = TableFunction(
        name="the_func", subquery=Select([tab.c["a"], tab.c["b"]], [tab])
    )
    sel = Select([Column("extra_column"), Column("*")], [func])
    result = sel.compile()
    expected = """\
SELECT
    "extra_column",
    *
FROM
    the_func((
        SELECT
            tab."a",
            tab."b"
        FROM
            tab
    ))"""
    assert result == expected


def test_select_star_and_aliased_column_from_table_returning_func():
    tab = Table(name="tab", columns=["a", "b"])
    func = TableFunction(
        name="the_func", subquery=Select([tab.c["a"], tab.c["b"]], [tab])
    )
    sel = Select([Column("extra_column", alias="extra"), Column("*")], [func])
    result = sel.compile()
    expected = """\
SELECT
    "extra_column" AS extra,
    *
FROM
    the_func((
        SELECT
            tab."a",
            tab."b"
        FROM
            tab
    ))"""
    assert result == expected


def test_select_table_returning_func_no_args():
    func = TableFunction(name="the_func")
    sel = Select([Column("*")], [func])
    result = sel.compile()
    expected = """\
SELECT
    *
FROM
    the_func()"""
    assert result == expected


def test_select_with_groupby():
    tab = Table(name="tab", columns=["a", "b"])
    func = ScalarFunction(name="the_func", columns=[tab.c["a"]])
    sel = Select([func], tables=[tab], groupby=[tab.c["b"]])
    expected = '''\
SELECT
    the_func(tab."a")
FROM
    tab
GROUP BY
    tab."b"'''
    result = sel.compile()
    assert result == expected


def test_select_with_groupby_aliased_table():
    tab = Table(name="tab", columns=["a", "b"], alias="the_table")
    func = ScalarFunction(name="the_func", columns=[tab.c["a"]])
    sel = Select([func], tables=[tab], groupby=[tab.c["b"]])
    expected = '''\
SELECT
    the_func(the_table."a")
FROM
    tab AS the_table
GROUP BY
    the_table."b"'''
    result = sel.compile()
    assert result == expected


def test_select_with_groupby_aliased_column():
    tab = Table(name="tab", columns=["a", "b"], alias="the_table")
    tab.c["b"].alias = "bbb"
    func = ScalarFunction(name="the_func", columns=[tab.c["a"]])
    sel = Select([func], tables=[tab], groupby=[tab.c["b"]])
    expected = '''\
SELECT
    the_func(the_table."a")
FROM
    tab AS the_table
GROUP BY
    the_table."b"'''
    result = sel.compile()
    assert result == expected


def test_select_with_orderby():
    tab = Table(name="tab", columns=["a", "b"])
    sel = Select([tab.c["a"], tab.c["b"]], [tab], orderby=[tab.c["a"], tab.c["b"]])
    expected = '''\
SELECT
    tab."a",
    tab."b"
FROM
    tab
ORDER BY
    "a",
    "b"'''
    result = sel.compile()
    assert expected == result
