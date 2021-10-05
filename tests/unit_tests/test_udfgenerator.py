# type: ignore
from typing import TypeVar

import pytest

from mipengine.udfgen.udfgenerator import (
    Column,
    IOType,
    LiteralArg,
    MergeTensorType,
    RelationArg,
    RelationType,
    ScalarFunction,
    Select,
    Table,
    TableFunction,
    TensorArg,
    TensorBinaryOp,
    UDFBadCall,
    UDFBadDefinition,
    convert_udfgenargs_to_udfargs,
    copy_types_from_udfargs,
    generate_udf_queries,
    get_funcparts_from_udf_registry,
    get_tensor_binary_op_template,
    get_matrix_transpose_template,
    get_udf_templates_using_udfregistry,
    literal,
    map_unknown_to_known_typeparams,
    mapping_inverse,
    mappings_coincide,
    merge_mappings_consistently,
    recursive_repr,
    relation,
    scalar,
    tensor,
    udf,
    verify_declared_typeparams_match_passed_type,
)
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import TableInfo
from mipengine.datatypes import DType
from mipengine.node_tasks_DTOs import TableSchema


@pytest.fixture(autouse=True)
def clear_udf_registry():
    yield
    udf.registry = {}


class TestUDFRegistry:
    @pytest.fixture(scope="class")
    def some_udfregistry(self):
        return {"some_function": lambda: 1}

    def test_get_func_from_udf_registry(self, some_udfregistry):
        funcname = "some_function"
        assert get_funcparts_from_udf_registry(funcname, some_udfregistry)() == 1

    def test_get_func_from_udf_registry_error(self, some_udfregistry):
        with pytest.raises(UDFBadCall) as exc:
            get_funcparts_from_udf_registry(
                funcname="not_there",
                udfregistry=some_udfregistry,
            )
        assert "cannot be found in udf registry" in str(exc)


def test_copy_types_from_udfargs():
    udfgen_args = {
        "a": RelationArg(table_name="A", schema=[]),
        "b": TensorArg(table_name="B", dtype=int, ndims=2),
    }
    udfparams = copy_types_from_udfargs(udfgen_args)
    assert udfparams == {
        "a": relation(schema=[]),
        "b": tensor(dtype=int, ndims=2),
    }


class TestVerifyTypeParams:
    @pytest.fixture(scope="class")
    def passed_type(self):
        class MockPassedType:
            a = 1
            b = 2
            c = 3

        return MockPassedType

    def test_verify_known_params_match_udfarg(self, passed_type):
        known_params = {"a": 1, "b": 2}
        assert (
            verify_declared_typeparams_match_passed_type(known_params, passed_type)
            is None
        )

    def test_verify_known_params_match_udfarg_missmatch_error(self, passed_type):
        known_params = {"a": 10, "b": 2}
        with pytest.raises(UDFBadCall):
            verify_declared_typeparams_match_passed_type(known_params, passed_type)

    def test_verify_known_params_match_udfarg_noattr_error(self, passed_type):
        known_params = {"A": 1, "b": 2}
        with pytest.raises(UDFBadCall):
            verify_declared_typeparams_match_passed_type(known_params, passed_type)


class TestTypeParamInference:
    def test_map_unknown_to_known_typeparams(self):
        unknown_params = {"a": 101, "b": 202}
        known_params = {"a": 1, "b": 2}
        inferred_typeparams = map_unknown_to_known_typeparams(
            unknown_params,
            known_params,
        )
        assert inferred_typeparams == {101: 1, 202: 2}

    def test_map_unknown_to_known_typeparams_error(self):
        unknown_params = {"d": 404, "b": 202}
        known_params = {"a": 1, "b": 2}
        with pytest.raises(ValueError):
            map_unknown_to_known_typeparams(
                unknown_params,
                known_params,
            )


class TestMergeMappings:
    def test_merge_mappings_no_overlap(self):
        mappings = [{"a": 1, "b": 2}, {"c": 3, "d": 4}]
        merged = merge_mappings_consistently(mappings)
        assert merged == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_merge_mappings_map_overlap(self):
        mappings = [{"a": 1, "b": 2}, {"c": 3, "a": 1}]
        merged = merge_mappings_consistently(mappings)
        assert merged == {"a": 1, "b": 2, "c": 3}

    def test_merge_mappings_error_missmatch(self):
        mappings = [{"a": 1, "b": 2}, {"c": 3, "a": 10}]
        with pytest.raises(ValueError):
            merge_mappings_consistently(mappings)


class TestMappingInverse:
    def test_mapping_inverse_valid(self):
        mapping = {"a": 1, "b": 2}
        assert mapping_inverse(mapping) == {1: "a", 2: "b"}

    def test_mapping_inverse_nvalid(self):
        mapping = {"a": 1, "b": 1}
        with pytest.raises(ValueError):
            mapping_inverse(mapping)


class TestUDFValidation:
    def test_validate_func_as_udf_invalid_input_type(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(x=tensor, return_type=scalar(int))
            def f_invalid_sig(x):
                x = 1
                return x

        assert "Input types of func are not subclasses of IOType" in str(exc)

    def test_validate_func_as_udf_invalid_output_type(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(x=tensor(int, 1), return_type=bool)
            def f_invalid_sig(x):
                x = 1
                return x

        assert "Output type of func is not subclass of IOType" in str(exc)

    def test_validate_func_as_udf_invalid_expression_in_return_stmt(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(x=tensor(int, 1), return_type=scalar(int))
            def f_invalid_ret_stmt(x):
                x = 1
                return x + 1

        assert "Expression in return statement" in str(exc)

    def test_validate_func_as_udf_invalid_no_return_stmt(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(x=tensor(int, 1), return_type=scalar(int))
            def f_invalid_ret_stmt(x):
                pass

        assert "Return statement not found" in str(exc)

    def test_validate_func_as_udf_invalid_parameter_names(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(y=tensor(int, 1), return_type=scalar(int))
            def f(x):
                return x

        assert "Invalid parameter names in udf decorator" in str(exc)

    def test_validate_func_as_udf_no_return_type(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(x=tensor(int, 1))
            def f(x):
                return x

        assert "Invalid parameter names in udf decorator" in str(exc)

    def test_validate_func_as_udf_valid(self):
        @udf(x=tensor(int, 1), return_type=scalar(int))
        def f(x):
            return x

        assert udf.registry != {}

    def test_tensors_and_relations(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(
                x=tensor(dtype=int, ndims=1),
                y=relation(schema=[]),
                return_type=scalar(int),
            )
            def f(x, y):
                return x

        assert "tensors and relations" in str(exc)


class TestMappingsCoincide:
    def test_mappings_coincide_true(self):
        map1 = {"a": 1, "b": 2}
        map2 = {"a": 1, "c": 3}
        assert mappings_coincide(map1, map2) is True

    def test_mappings_coincide_false(self):
        map1 = {"a": 1, "b": 2}
        map2 = {"a": 10, "c": 3}
        assert mappings_coincide(map1, map2) is False


def test_constructor_repr():
    class A:
        def __init__(self, a, b) -> None:
            self.a, self.b = a, b

        __repr__ = recursive_repr

    a = A(1, 2)
    assert repr(a) == "A(a=1,b=2)"


def test_tensor_generic():
    DT = TypeVar("DT")
    t = tensor(dtype=DT, ndims=2)
    assert t.is_generic
    assert t.known_typeparams
    assert t.unknown_typeparams


def test_tensor_specific():
    t = tensor(dtype=int, ndims=3)
    assert not t.is_generic
    assert t.known_typeparams
    assert not t.unknown_typeparams


def test_relation_generic():
    S = TypeVar("S")
    r = relation(schema=S)
    assert r.is_generic
    assert not r.known_typeparams
    assert r.unknown_typeparams


def test_relation_specific():
    r = relation(schema=[("c1", int), ("c2", float)])
    assert isinstance(r, RelationType)


def test_literal():
    ltr = literal()
    assert isinstance(ltr, IOType)


def test_scalar():
    s = scalar(dtype=int)
    assert not s.is_generic


def test_tensor_schema():
    t = tensor(dtype=DType.FLOAT, ndims=2)
    assert t.schema == [("dim0", DType.INT), ("dim1", DType.INT), ("val", DType.FLOAT)]


def test_tensor_column_names():
    t = tensor(dtype=DType.FLOAT, ndims=2)
    colnames = t.column_names(prefix="pre")
    assert colnames == ["pre_dim0", "pre_dim1", "pre_val"]


def test_relation_column_names():
    r = relation(schema=[("ci", DType.INT), ("cf", DType.FLOAT), ("cs", DType.STR)])
    colnames = r.column_names(prefix="pre")
    assert colnames == ["pre_ci", "pre_cf", "pre_cs"]


def test_udf_decorator_already_there():
    @udf(return_type=scalar(int))
    def already_there():
        x = 1
        return x

    with pytest.raises(UDFBadDefinition) as exc:

        @udf(return_type=scalar(int))
        def already_there():
            x = 1
            return x

    assert "already in the udf registry" in str(exc)


def test_convert_udfgenargs_to_udfargs_relation():
    udfgen_posargs = [
        TableInfo(
            name="tab",
            schema_=TableSchema(
                columns=[
                    ColumnInfo(name="c1", dtype=DType.INT),
                    ColumnInfo(name="c2", dtype=DType.FLOAT),
                    ColumnInfo(name="c3", dtype=DType.STR),
                ]
            ),
        )
    ]
    expected_udf_posargs = [
        RelationArg(table_name="tab", schema=[("c1", int), ("c2", float), ("c3", str)])
    ]
    result, _ = convert_udfgenargs_to_udfargs(udfgen_posargs, {})
    assert result == expected_udf_posargs


def test_convert_udfgenargs_to_udfargs_tensor():
    udfgen_posargs = [
        TableInfo(
            name="tab",
            schema_=TableSchema(
                columns=[
                    ColumnInfo(name="node_id", dtype=DType.STR),
                    ColumnInfo(name="dim0", dtype=DType.INT),
                    ColumnInfo(name="dim1", dtype=DType.INT),
                    ColumnInfo(name="val", dtype=DType.FLOAT),
                ]
            ),
        )
    ]
    expected_udf_posargs = [TensorArg(table_name="tab", dtype=float, ndims=2)]
    result, _ = convert_udfgenargs_to_udfargs(udfgen_posargs, {})
    assert result == expected_udf_posargs


def test_convert_udfgenargs_to_udfargs_literal():
    udfgen_posargs = [42]
    expected_udf_posargs = [LiteralArg(value=42)]
    result, _ = convert_udfgenargs_to_udfargs(udfgen_posargs, {})
    assert result == expected_udf_posargs


def test_convert_udfgenargs_to_udfargs_multiple_types():
    udfgen_posargs = [
        TableInfo(
            name="tab",
            schema_=TableSchema(
                columns=[
                    ColumnInfo(name="c1", dtype=DType.INT),
                    ColumnInfo(name="c2", dtype=DType.FLOAT),
                    ColumnInfo(name="c3", dtype=DType.STR),
                ]
            ),
        ),
        TableInfo(
            name="tab",
            schema_=TableSchema(
                columns=[
                    ColumnInfo(name="node_id", dtype=DType.STR),
                    ColumnInfo(name="dim0", dtype=DType.INT),
                    ColumnInfo(name="dim1", dtype=DType.INT),
                    ColumnInfo(name="val", dtype=DType.FLOAT),
                ]
            ),
        ),
        42,
    ]
    expected_udf_posargs = [
        RelationArg(table_name="tab", schema=[("c1", int), ("c2", float), ("c3", str)]),
        TensorArg(table_name="tab", dtype=float, ndims=2),
        LiteralArg(value=42),
    ]
    result, _ = convert_udfgenargs_to_udfargs(udfgen_posargs, {})
    assert result == expected_udf_posargs


class TestTensorArgsEquality:
    def test_tensor_args_names_not_equal(self):
        t1 = TensorArg("a", int, 1)
        t2 = TensorArg("b", int, 1)
        assert t1 != t2

    def test_tensor_args_dtypes_not_equal(self):
        t1 = TensorArg("a", int, 1)
        t2 = TensorArg("a", float, 1)
        assert t1 != t2

    def test_tensor_args_ndims_not_equal(self):
        t1 = TensorArg("a", int, 1)
        t2 = TensorArg("a", int, 2)
        assert t1 != t2

    def test_tensor_args_equal(self):
        t1 = TensorArg("a", int, 1)
        t2 = TensorArg("a", int, 1)
        assert t1 == t2

    def test_tensor_args_equal_different_dtype_representation(self):
        t1 = TensorArg("a", int, 1)
        t2 = TensorArg("a", DType.INT, 1)
        assert t1 == t2


class TestRelationArgsEquality:
    def test_relation_args_names_not_equal(self):
        t1 = RelationArg("a", [])
        t2 = RelationArg("b", [])
        assert t1 != t2

    def test_relation_args_schemata_names_not_equal(self):
        t1 = RelationArg("a", [("c1", int), ("c2", int)])
        t2 = RelationArg("a", [("c10", int), ("c2", int)])
        assert t1 != t2

    def test_relation_args_schemata_dtypes_not_equal(self):
        t1 = RelationArg("a", [("c1", int), ("c2", int)])
        t2 = RelationArg("a", [("c1", int), ("c2", float)])
        assert t1 != t2

    def test_relation_args_schemata_lenghts_not_equal(self):
        t1 = RelationArg("a", [("c1", int), ("c2", int)])
        t2 = RelationArg("a", [("c1", int)])
        assert t1 != t2

    def test_relation_args_equal(self):
        t1 = RelationArg("a", [("c1", int), ("c2", int)])
        t2 = RelationArg("a", [("c1", int), ("c2", int)])
        assert t1 == t2


# ~~~~~~~~~~~~~~~~~~~~~~ Test Select AST Node ~~~~~~~~~~~~~~~~ #


def test_column_alias():
    col = Column("very_long_name", alias="name")
    result = col.compile()
    expected = "very_long_name AS name"
    assert result == expected


def test_column_mul():
    col1 = Column("col1", alias="column1")
    col2 = Column("col2", alias="column2")
    prod = col1 * col2
    result = prod.compile()
    expected = "col1 * col2"
    assert result == expected


def test_column_add():
    col1 = Column("col1", alias="column1")
    col2 = Column("col2", alias="column2")
    prod = col1 + col2
    result = prod.compile()
    expected = "col1 + col2"
    assert result == expected


def test_column_sub():
    col1 = Column("col1", alias="column1")
    col2 = Column("col2", alias="column2")
    prod = col1 - col2
    result = prod.compile()
    expected = "col1 - col2"
    assert result == expected


def test_column_div():
    col1 = Column("col1", alias="column1")
    col2 = Column("col2", alias="column2")
    prod = col1 / col2
    result = prod.compile()
    expected = "col1 / col2"
    assert result == expected


def test_column_mul_from_table():
    tab = Table(name="tab", columns=["a", "b"])
    prod = tab.c["a"] * tab.c["b"]
    result = prod.compile()
    expected = "tab.a * tab.b"
    assert result == expected


def test_select_single_table():
    tab = Table(name="a_table", columns=["c1", "c2"])
    sel = Select([tab.c["c1"], tab.c["c2"]], [tab])
    result = sel.compile()
    expected = """\
SELECT
    a_table.c1,
    a_table.c2
FROM
    a_table"""
    assert result == expected


def test_select_two_tables_joined():
    tab1 = Table(name="tab1", columns=["a", "b"])
    tab2 = Table(name="tab2", columns=["c", "d"])
    sel = Select([tab1.c["a"], tab1.c["b"], tab2.c["c"], tab2.c["d"]], [tab1, tab2])
    expected = """\
SELECT
    tab1.a,
    tab1.b,
    tab2.c,
    tab2.d
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
    expected = """\
SELECT
    tab1.a,
    tab1.b,
    tab2.c
FROM
    tab1,
    tab2
WHERE
    tab1.b=tab2.b"""
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
    expected = """\
SELECT
    tab1.a,
    tab1.b,
    tab1.c,
    tab2.d
FROM
    tab1,
    tab2
WHERE
    tab1.b=tab2.b AND
    tab1.c=tab2.c"""
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
    the_func(tab1.a,tab1.b)
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
            tab.a,
            tab.b
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
    extra_column,
    *
FROM
    the_func((
        SELECT
            tab.a,
            tab.b
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
    extra_column AS extra,
    *
FROM
    the_func((
        SELECT
            tab.a,
            tab.b
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
    expected = """\
SELECT
    the_func(tab.a)
FROM
    tab
GROUP BY
    tab.b"""
    result = sel.compile()
    assert result == expected


def test_select_with_groupby_aliased_table():
    tab = Table(name="tab", columns=["a", "b"], alias="the_table")
    func = ScalarFunction(name="the_func", columns=[tab.c["a"]])
    sel = Select([func], tables=[tab], groupby=[tab.c["b"]])
    expected = """\
SELECT
    the_func(the_table.a)
FROM
    tab AS the_table
GROUP BY
    the_table.b"""
    result = sel.compile()
    assert result == expected


def test_select_with_groupby_aliased_column():
    tab = Table(name="tab", columns=["a", "b"], alias="the_table")
    tab.c["b"].alias = "bbb"
    func = ScalarFunction(name="the_func", columns=[tab.c["a"]])
    sel = Select([func], tables=[tab], groupby=[tab.c["b"]])
    expected = """\
SELECT
    the_func(the_table.a)
FROM
    tab AS the_table
GROUP BY
    the_table.b"""
    result = sel.compile()
    assert result == expected


def test_select_with_orderby():
    tab = Table(name="tab", columns=["a", "b"])
    sel = Select([tab.c["a"], tab.c["b"]], [tab], orderby=[tab.c["a"], tab.c["b"]])
    expected = """\
SELECT
    tab.a,
    tab.b
FROM
    tab
ORDER BY
    a,
    b"""
    result = sel.compile()
    assert expected == result


# ~~~~~~~~~~~~~~~~~~~~~~ Test UDF Generator ~~~~~~~~~~~~~~~~~~ #


class TestUDFGenBase:
    @pytest.fixture(scope="class")
    def funcname(self, udfregistry):
        assert len(udfregistry) == 1
        return next(iter(udfregistry.keys()))


class TestUDFGen_InvalidUDFArgs_NamesMismatch(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            x=tensor(dtype=int, ndims=1),
            y=tensor(dtype=int, ndims=1),
            z=literal(),
            return_type=scalar(int),
        )
        def f(x, y, z):
            return x

        return udf.registry

    def test_get_udf_templates(self, udfregistry, funcname):
        posargs = [TensorArg("table_name", dtype=int, ndims=1)]
        keywordargs = {"z": LiteralArg(1)}
        with pytest.raises(UDFBadCall) as exc:
            udfdef, udfsel = get_udf_templates_using_udfregistry(
                funcname, posargs, keywordargs, udfregistry
            )
        assert "UDF argument names do not match UDF parameter names" in str(exc)


class TestUDFGen_InvalidUDFArgs_InconsistentTypeVars(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        T = TypeVar("T")

        @udf(
            x=tensor(dtype=T, ndims=1),
            y=tensor(dtype=T, ndims=1),
            return_type=scalar(T),
        )
        def f(x, y):
            return x

        return udf.registry

    def test_get_udf_templates(self, udfregistry, funcname):
        posargs = [
            TensorArg("table_name1", dtype=int, ndims=1),
            TensorArg("table_name1", dtype=float, ndims=1),
        ]
        keywordargs = {}
        with pytest.raises(ValueError) as e:
            udfdef, udfsel = get_udf_templates_using_udfregistry(
                funcname,
                posargs,
                keywordargs,
                udfregistry,
            )
        err_msg, *_ = e.value.args
        assert "inconsistent mappings" in err_msg


class TestUDFGen_TensorToTensor(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        T = TypeVar("T")

        @udf(x=tensor(dtype=T, ndims=2), return_type=tensor(dtype=DType.FLOAT, ndims=2))
        def f(x):
            result = x
            return result

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="tensor_in_db",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="node_id", dtype=DType.STR),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="dim1", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
            )
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name(x_dim0 INT,x_dim1 INT,x_val INT)
RETURNS
TABLE(dim0 INT,dim1 INT,val REAL)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    x = udfio.from_tensor_table({n: _columns[n] for n in ['x_dim0', 'x_dim1', 'x_val']})
    result = x
    return udfio.as_tensor_table(numpy.array(result))
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
DROP TABLE IF EXISTS $table_name;
CREATE TABLE $table_name(node_id VARCHAR(50),dim0 INT,dim1 INT,val REAL);
INSERT INTO $table_name
SELECT
    CAST('$node_id' AS VARCHAR(50)) AS node_id,
    *
FROM
    $udf_name((
        SELECT
            tensor_in_db.dim0,
            tensor_in_db.dim1,
            tensor_in_db.val
        FROM
            tensor_in_db
    ));"""

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfsel,
    ):
        udfdef, udfsel = generate_udf_queries(funcname, positional_args, {})
        assert udfdef.template == expected_udfdef
        assert udfsel.template == expected_udfsel


class TestUDFGen_RelationToTensor(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        S = TypeVar("S")

        @udf(r=relation(schema=S), return_type=tensor(dtype=DType.FLOAT, ndims=2))
        def f(r):
            result = r
            return result

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="rel_in_db",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="col0", dtype=DType.INT),
                        ColumnInfo(name="col1", dtype=DType.FLOAT),
                        ColumnInfo(name="col2", dtype=DType.STR),
                    ]
                ),
            )
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name(r_col0 INT,r_col1 REAL,r_col2 VARCHAR(50))
RETURNS
TABLE(dim0 INT,dim1 INT,val REAL)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    r = pd.DataFrame({n: _columns[n] for n in ['r_col0', 'r_col1', 'r_col2']})
    result = r
    return udfio.as_tensor_table(numpy.array(result))
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
DROP TABLE IF EXISTS $table_name;
CREATE TABLE $table_name(node_id VARCHAR(50),dim0 INT,dim1 INT,val REAL);
INSERT INTO $table_name
SELECT
    CAST('$node_id' AS VARCHAR(50)) AS node_id,
    *
FROM
    $udf_name((
        SELECT
            rel_in_db.col0,
            rel_in_db.col1,
            rel_in_db.col2
        FROM
            rel_in_db
    ));"""

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfsel,
    ):
        udfdef, udfsel = generate_udf_queries(funcname, positional_args, {})
        assert udfdef.template == expected_udfdef
        assert udfsel.template == expected_udfsel


class TestUDFGen_TensorToRelation(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            x=tensor(dtype=int, ndims=1),
            return_type=relation(schema=[("ci", int), ("cf", float)]),
        )
        def f(x):
            return x

        return udf.registry

    @pytest.fixture(scope="class")
    def udf_args(self):
        return {"x": TensorArg(table_name="tensor_in_db", dtype=int, ndims=1)}

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="tensor_in_db",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="node_id", dtype=DType.STR),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
            )
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name(x_dim0 INT,x_val INT)
RETURNS
TABLE(ci INT,cf REAL)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    x = udfio.from_tensor_table({n: _columns[n] for n in ['x_dim0', 'x_val']})
    return udfio.as_relational_table(numpy.array(x))
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
DROP TABLE IF EXISTS $table_name;
CREATE TABLE $table_name(node_id VARCHAR(50),ci INT,cf REAL);
INSERT INTO $table_name
SELECT
    CAST('$node_id' AS VARCHAR(50)) AS node_id,
    *
FROM
    $udf_name((
        SELECT
            tensor_in_db.dim0,
            tensor_in_db.val
        FROM
            tensor_in_db
    ));"""

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfsel,
    ):
        udfdef, udfsel = generate_udf_queries(funcname, positional_args, {})
        assert udfdef.template == expected_udfdef
        assert udfsel.template == expected_udfsel


class TestUDFGen_LiteralArgument(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            x=tensor(dtype=DType.INT, ndims=1),
            v=literal(),
            return_type=scalar(dtype=DType.INT),
        )
        def f(x, v):
            result = v
            return result

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="the_table",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.STR),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
            ),
            42,
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name(x_dim0 INT,x_val INT)
RETURNS
INT
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    x = udfio.from_tensor_table({n: _columns[n] for n in ['x_dim0', 'x_val']})
    v = 42
    result = v
    return result
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
DROP TABLE IF EXISTS $table_name;
CREATE TABLE $table_name(result INT);
INSERT INTO $table_name
SELECT
    $udf_name(the_table.dim0,the_table.val)
FROM
    the_table;"""

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfsel,
    ):
        udfdef, udfsel = generate_udf_queries(funcname, positional_args, {})
        assert udfdef.template == expected_udfdef
        assert udfsel.template == expected_udfsel


class TestUDFGen_ManyLiteralArguments(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            x=tensor(dtype=DType.INT, ndims=1),
            v=literal(),
            w=literal(),
            return_type=scalar(dtype=DType.INT),
        )
        def f(x, v, w):
            result = v + w
            return result

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="tensor_in_db",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.STR),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
            ),
            42,
            24,
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name(x_dim0 INT,x_val INT)
RETURNS
INT
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    x = udfio.from_tensor_table({n: _columns[n] for n in ['x_dim0', 'x_val']})
    v = 42
    w = 24
    result = v + w
    return result
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
DROP TABLE IF EXISTS $table_name;
CREATE TABLE $table_name(result INT);
INSERT INTO $table_name
SELECT
    $udf_name(tensor_in_db.dim0,tensor_in_db.val)
FROM
    tensor_in_db;"""

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfsel,
    ):
        udfdef, udfsel = generate_udf_queries(funcname, positional_args, {})
        assert udfdef.template == expected_udfdef
        assert udfsel.template == expected_udfsel


class TestUDFGen_NoArguments(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            return_type=tensor(dtype=int, ndims=1),
        )
        def f():
            x = [1, 2, 3]
            return x

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return []

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name()
RETURNS
TABLE(dim0 INT,val INT)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    x = [1, 2, 3]
    return udfio.as_tensor_table(numpy.array(x))
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
DROP TABLE IF EXISTS $table_name;
CREATE TABLE $table_name(node_id VARCHAR(50),dim0 INT,val INT);
INSERT INTO $table_name
SELECT
    CAST('$node_id' AS VARCHAR(50)) AS node_id,
    *
FROM
    $udf_name();"""

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfsel,
    ):
        udfdef, udfsel = generate_udf_queries(funcname, positional_args, {})
        assert udfdef.template == expected_udfdef
        assert udfsel.template == expected_udfsel


class TestUDFGen_RelationInExcludeRowId(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        S = TypeVar("S")

        @udf(r=relation(schema=S), return_type=tensor(dtype=DType.FLOAT, ndims=2))
        def f(r):
            result = r
            return result

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="rel_in_db",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.STR),
                        ColumnInfo(name="c0", dtype=DType.INT),
                        ColumnInfo(name="c1", dtype=DType.FLOAT),
                        ColumnInfo(name="c2", dtype=DType.STR),
                    ]
                ),
            )
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name(r_c0 INT,r_c1 REAL,r_c2 VARCHAR(50))
RETURNS
TABLE(dim0 INT,dim1 INT,val REAL)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    r = pd.DataFrame({n: _columns[n] for n in ['r_c0', 'r_c1', 'r_c2']})
    result = r
    return udfio.as_tensor_table(numpy.array(result))
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
DROP TABLE IF EXISTS $table_name;
CREATE TABLE $table_name(node_id VARCHAR(50),dim0 INT,dim1 INT,val REAL);
INSERT INTO $table_name
SELECT
    CAST('$node_id' AS VARCHAR(50)) AS node_id,
    *
FROM
    $udf_name((
        SELECT
            rel_in_db.c0,
            rel_in_db.c1,
            rel_in_db.c2
        FROM
            rel_in_db
    ));"""

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfsel,
    ):
        udfdef, udfsel = generate_udf_queries(funcname, positional_args, {})
        assert udfdef.template == expected_udfdef
        assert udfsel.template == expected_udfsel


class TestUDFGen_UnknownReturnDimensions(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        T = TypeVar("S")
        N = TypeVar("N")

        @udf(t=tensor(dtype=T, ndims=N), return_type=tensor(dtype=T, ndims=N))
        def f(t):
            result = t
            return result

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="tens_in_db",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.STR),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="dim1", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
            )
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name(t_dim0 INT,t_dim1 INT,t_val INT)
RETURNS
TABLE(dim0 INT,dim1 INT,val INT)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    t = udfio.from_tensor_table({n: _columns[n] for n in ['t_dim0', 't_dim1', 't_val']})
    result = t
    return udfio.as_tensor_table(numpy.array(result))
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
DROP TABLE IF EXISTS $table_name;
CREATE TABLE $table_name(node_id VARCHAR(50),dim0 INT,dim1 INT,val INT);
INSERT INTO $table_name
SELECT
    CAST('$node_id' AS VARCHAR(50)) AS node_id,
    *
FROM
    $udf_name((
        SELECT
            tens_in_db.dim0,
            tens_in_db.dim1,
            tens_in_db.val
        FROM
            tens_in_db
    ));"""

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfsel,
    ):
        udfdef, udfsel = generate_udf_queries(funcname, positional_args, {})
        assert udfdef.template == expected_udfdef
        assert udfsel.template == expected_udfsel


class TestUDFGen_TwoTensors1DReturnTable(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            x=tensor(dtype=int, ndims=1),
            y=tensor(dtype=int, ndims=1),
            return_type=tensor(dtype=int, ndims=1),
        )
        def f(x, y):
            result = x - y
            return result

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="tens0",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.STR),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
            ),
            TableInfo(
                name="tens1",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.STR),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name(x_dim0 INT,x_val INT,y_dim0 INT,y_val INT)
RETURNS
TABLE(dim0 INT,val INT)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    x = udfio.from_tensor_table({n: _columns[n] for n in ['x_dim0', 'x_val']})
    y = udfio.from_tensor_table({n: _columns[n] for n in ['y_dim0', 'y_val']})
    result = x - y
    return udfio.as_tensor_table(numpy.array(result))
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
DROP TABLE IF EXISTS $table_name;
CREATE TABLE $table_name(node_id VARCHAR(50),dim0 INT,val INT);
INSERT INTO $table_name
SELECT
    CAST('$node_id' AS VARCHAR(50)) AS node_id,
    *
FROM
    $udf_name((
        SELECT
            tens0.dim0,
            tens0.val,
            tens1.dim0,
            tens1.val
        FROM
            tens0,
            tens1
        WHERE
            tens0.dim0=tens1.dim0
    ));"""

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfsel,
    ):
        udfdef, udfsel = generate_udf_queries(funcname, positional_args, {})
        assert udfdef.template == expected_udfdef
        assert udfsel.template == expected_udfsel


class TestUDFGen_ThreeTensors1DReturnTable(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            x=tensor(dtype=int, ndims=1),
            y=tensor(dtype=int, ndims=1),
            z=tensor(dtype=int, ndims=1),
            return_type=tensor(dtype=int, ndims=1),
        )
        def f(x, y, z):
            result = x + y + z
            return result

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="tens0",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.STR),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
            ),
            TableInfo(
                name="tens1",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.STR),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
            ),
            TableInfo(
                name="tens2",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.STR),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name(x_dim0 INT,x_val INT,y_dim0 INT,y_val INT,z_dim0 INT,z_val INT)
RETURNS
TABLE(dim0 INT,val INT)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    x = udfio.from_tensor_table({n: _columns[n] for n in ['x_dim0', 'x_val']})
    y = udfio.from_tensor_table({n: _columns[n] for n in ['y_dim0', 'y_val']})
    z = udfio.from_tensor_table({n: _columns[n] for n in ['z_dim0', 'z_val']})
    result = x + y + z
    return udfio.as_tensor_table(numpy.array(result))
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
DROP TABLE IF EXISTS $table_name;
CREATE TABLE $table_name(node_id VARCHAR(50),dim0 INT,val INT);
INSERT INTO $table_name
SELECT
    CAST('$node_id' AS VARCHAR(50)) AS node_id,
    *
FROM
    $udf_name((
        SELECT
            tens0.dim0,
            tens0.val,
            tens1.dim0,
            tens1.val,
            tens2.dim0,
            tens2.val
        FROM
            tens0,
            tens1,
            tens2
        WHERE
            tens0.dim0=tens1.dim0 AND
            tens0.dim0=tens2.dim0
    ));"""

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfsel,
    ):
        udfdef, udfsel = generate_udf_queries(funcname, positional_args, {})
        assert udfdef.template == expected_udfdef
        assert udfsel.template == expected_udfsel


class TestUDFGen_ThreeTensors2DReturnTable(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            x=tensor(dtype=int, ndims=2),
            y=tensor(dtype=int, ndims=2),
            z=tensor(dtype=int, ndims=2),
            return_type=tensor(dtype=int, ndims=2),
        )
        def f(x, y, z):
            result = x + y + z
            return result

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="tens0",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.STR),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="dim1", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
            ),
            TableInfo(
                name="tens1",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.STR),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="dim1", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
            ),
            TableInfo(
                name="tens2",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.STR),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="dim1", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name(x_dim0 INT,x_dim1 INT,x_val INT,y_dim0 INT,y_dim1 INT,y_val INT,z_dim0 INT,z_dim1 INT,z_val INT)
RETURNS
TABLE(dim0 INT,dim1 INT,val INT)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    x = udfio.from_tensor_table({n: _columns[n] for n in ['x_dim0', 'x_dim1', 'x_val']})
    y = udfio.from_tensor_table({n: _columns[n] for n in ['y_dim0', 'y_dim1', 'y_val']})
    z = udfio.from_tensor_table({n: _columns[n] for n in ['z_dim0', 'z_dim1', 'z_val']})
    result = x + y + z
    return udfio.as_tensor_table(numpy.array(result))
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
DROP TABLE IF EXISTS $table_name;
CREATE TABLE $table_name(node_id VARCHAR(50),dim0 INT,dim1 INT,val INT);
INSERT INTO $table_name
SELECT
    CAST('$node_id' AS VARCHAR(50)) AS node_id,
    *
FROM
    $udf_name((
        SELECT
            tens0.dim0,
            tens0.dim1,
            tens0.val,
            tens1.dim0,
            tens1.dim1,
            tens1.val,
            tens2.dim0,
            tens2.dim1,
            tens2.val
        FROM
            tens0,
            tens1,
            tens2
        WHERE
            tens0.dim0=tens1.dim0 AND
            tens0.dim1=tens1.dim1 AND
            tens0.dim0=tens2.dim0 AND
            tens0.dim1=tens2.dim1
    ));"""

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfsel,
    ):
        udfdef, udfsel = generate_udf_queries(funcname, positional_args, {})
        assert udfdef.template == expected_udfdef
        assert udfsel.template == expected_udfsel


class TestUDFGen_TwoTensors1DReturnScalar(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        T = TypeVar("T")
        N = TypeVar("N")

        @udf(x=tensor(T, N), y=tensor(T, N), return_type=scalar(float))
        def f(x, y):
            result = sum(x - y)
            return result

        return udf.registry

    @pytest.fixture(scope="class")
    def udf_args(self):
        return {
            "x": TensorArg(table_name="tens0", dtype=int, ndims=1),
            "y": TensorArg(table_name="tens1", dtype=int, ndims=1),
        }

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="tens0",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.STR),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
            ),
            TableInfo(
                name="tens1",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.STR),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name(x_dim0 INT,x_val INT,y_dim0 INT,y_val INT)
RETURNS
REAL
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    x = udfio.from_tensor_table({n: _columns[n] for n in ['x_dim0', 'x_val']})
    y = udfio.from_tensor_table({n: _columns[n] for n in ['y_dim0', 'y_val']})
    result = sum(x - y)
    return result
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
DROP TABLE IF EXISTS $table_name;
CREATE TABLE $table_name(result REAL);
INSERT INTO $table_name
SELECT
    $udf_name(tens0.dim0,tens0.val,tens1.dim0,tens1.val)
FROM
    tens0,
    tens1
WHERE
    tens0.dim0=tens1.dim0;"""

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfsel,
    ):
        udfdef, udfsel = generate_udf_queries(funcname, positional_args, {})
        assert udfdef.template == expected_udfdef
        assert udfsel.template == expected_udfsel


class TestUDFGen_SQLTensorMultOut1D(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def funcname(self):
        return TensorBinaryOp.MATMUL.name

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="tensor1",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="node_id", dtype=DType.STR),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="dim1", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
            ),
            TableInfo(
                name="tensor2",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="node_id", dtype=DType.STR),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return ""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
DROP TABLE IF EXISTS $table_name;
CREATE TABLE $table_name(node_id VARCHAR(50),dim0 INT,val REAL);
INSERT INTO $table_name
SELECT
    CAST('$node_id' AS VARCHAR(50)) AS node_id,
    tensor_0.dim0 AS dim0,
    SUM(tensor_0.val * tensor_1.val) AS val
FROM
    tensor1 AS tensor_0,
    tensor2 AS tensor_1
WHERE
    tensor_0.dim1=tensor_1.dim0
GROUP BY
    tensor_0.dim0
ORDER BY
    dim0;"""

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfsel,
    ):
        udfdef, udfsel = generate_udf_queries(funcname, positional_args, {})
        assert udfdef.template == expected_udfdef
        assert udfsel.template == expected_udfsel


class TestUDFGen_SQLTensorMultOut2D(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def funcname(self):
        return TensorBinaryOp.MATMUL.name

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="tensor1",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="node_id", dtype=DType.STR),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="dim1", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
            ),
            TableInfo(
                name="tensor2",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="node_id", dtype=DType.STR),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="dim1", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return ""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
DROP TABLE IF EXISTS $table_name;
CREATE TABLE $table_name(node_id VARCHAR(50),dim0 INT,dim1 INT,val REAL);
INSERT INTO $table_name
SELECT
    CAST('$node_id' AS VARCHAR(50)) AS node_id,
    tensor_0.dim0 AS dim0,
    tensor_1.dim1 AS dim1,
    SUM(tensor_0.val * tensor_1.val) AS val
FROM
    tensor1 AS tensor_0,
    tensor2 AS tensor_1
WHERE
    tensor_0.dim1=tensor_1.dim0
GROUP BY
    tensor_0.dim0,
    tensor_1.dim1
ORDER BY
    dim0,
    dim1;"""

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfsel,
    ):
        udfdef, udfsel = generate_udf_queries(funcname, positional_args, {})
        assert udfdef.template == expected_udfdef
        assert udfsel.template == expected_udfsel


class TestUDFGen_SQLTensorSubLiteralArg(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def funcname(self):
        return TensorBinaryOp.SUB.name

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            1,
            TableInfo(
                name="tensor1",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="node_id", dtype=DType.STR),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return ""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
DROP TABLE IF EXISTS $table_name;
CREATE TABLE $table_name(node_id VARCHAR(50),dim0 INT,val REAL);
INSERT INTO $table_name
SELECT
    CAST('$node_id' AS VARCHAR(50)) AS node_id,
    tensor_0.dim0 AS dim0,
    1 - tensor_0.val AS val
FROM
    tensor1 AS tensor_0;"""

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfsel,
    ):
        udfdef, udfsel = generate_udf_queries(funcname, positional_args, {})
        assert udfdef.template == expected_udfdef
        assert udfsel.template == expected_udfsel


class TestUDFGen_ScalarReturn(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        T = TypeVar("T")

        @udf(x=tensor(dtype=T, ndims=2), return_type=scalar(dtype=T))
        def f(x):
            result = sum(x)
            return result

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="tensor_in_db",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.STR),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="dim1", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
            )
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name(x_dim0 INT,x_dim1 INT,x_val INT)
RETURNS
INT
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    x = udfio.from_tensor_table({n: _columns[n] for n in ['x_dim0', 'x_dim1', 'x_val']})
    result = sum(x)
    return result
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
DROP TABLE IF EXISTS $table_name;
CREATE TABLE $table_name(result INT);
INSERT INTO $table_name
SELECT
    $udf_name(tensor_in_db.dim0,tensor_in_db.dim1,tensor_in_db.val)
FROM
    tensor_in_db;"""

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfsel,
    ):
        udfdef, udfsel = generate_udf_queries(funcname, positional_args, {})
        assert udfdef.template == expected_udfdef
        assert udfsel.template == expected_udfsel


class TestUDFGen_MergeTensor(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            xs=MergeTensorType(dtype=int, ndims=1),
            return_type=tensor(dtype=int, ndims=1),
        )
        def sum_tensors(xs):
            x = sum(xs)
            return x

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="merge_table",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.STR),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
            )
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name(xs_node_id VARCHAR(50),xs_dim0 INT,xs_val INT)
RETURNS
TABLE(dim0 INT,val INT)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    xs = udfio.merge_tensor_to_list({n: _columns[n] for n in ['xs_node_id', 'xs_dim0', 'xs_val']})
    x = sum(xs)
    return udfio.as_tensor_table(numpy.array(x))
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
DROP TABLE IF EXISTS $table_name;
CREATE TABLE $table_name(node_id VARCHAR(50),dim0 INT,val INT);
INSERT INTO $table_name
SELECT
    CAST('$node_id' AS VARCHAR(50)) AS node_id,
    *
FROM
    $udf_name((
        SELECT
            merge_table.node_id,
            merge_table.dim0,
            merge_table.val
        FROM
            merge_table
    ));"""

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfsel,
    ):
        udfdef, udfsel = generate_udf_queries(funcname, positional_args, {})
        assert udfdef.template == expected_udfdef
        assert udfsel.template == expected_udfsel


class TestUDFGen_TracebackFlag(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(x=tensor(dtype=int, ndims=1), return_type=tensor(dtype=int, ndims=1))
        def f(x):
            y = x + 1
            z = 1 / 0
            return z

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="tensor_in_db",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="node_id", dtype=DType.STR),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
            )
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return r"""CREATE OR REPLACE FUNCTION
$udf_name(x_dim0 INT,x_val INT)
RETURNS
VARCHAR(50)
LANGUAGE PYTHON
{
    __code = ['import pandas as pd', 'import udfio', "x = udfio.from_tensor_table({n: _columns[n] for n in ['x_dim0', 'x_val']})", 'y = x + 1', 'z = 1 / 0']
    import traceback
    try:
        import pandas as pd
        import udfio
        x = udfio.from_tensor_table({n: _columns[n] for n in ['x_dim0', 'x_val']})
        y = x + 1
        z = 1 / 0
        return "no error"
    except Exception as e:
        offset = 5
        tb = e.__traceback__
        lineno = tb.tb_lineno - offset
        line = ' ' * 4 + __code[lineno]
        linelen = len(__code[lineno])
        underline = ' ' * 4 + '^' * linelen
        tb_lines = traceback.format_tb(tb)
        tb_lines.insert(1, line)
        tb_lines.insert(2, underline)
        tb_lines.append(repr(e))
        tb_formatted = '\n'.join(tb_lines)
        return tb_formatted
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
DROP TABLE IF EXISTS $table_name;
CREATE TABLE $table_name(result VARCHAR(50));
INSERT INTO $table_name
SELECT
    $udf_name(tensor_in_db.dim0,tensor_in_db.val)
FROM
    tensor_in_db;"""

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfsel,
    ):
        udfdef, udfsel = generate_udf_queries(
            funcname, positional_args, {}, traceback=True
        )
        assert udfdef.template == expected_udfdef
        assert udfsel.template == expected_udfsel


# ~~~~~~~~~~~~~~~~~~~~~~ Test SQL Generator ~~~~~~~~~~~~~~~~~~ #


def test_tensor_elementwise_binary_op_1dim():
    tensor_0 = TensorArg(table_name="tens0", dtype=None, ndims=1)
    tensor_1 = TensorArg(table_name="tens1", dtype=None, ndims=1)
    op = TensorBinaryOp.ADD
    expected = """\
SELECT
    CAST('$node_id' AS VARCHAR(50)) AS node_id,
    tensor_0.dim0 AS dim0,
    tensor_0.val + tensor_1.val AS val
FROM
    tens0 AS tensor_0,
    tens1 AS tensor_1
WHERE
    tensor_0.dim0=tensor_1.dim0"""
    result = get_tensor_binary_op_template(tensor_0, tensor_1, op)
    assert result == expected


def test_tensor_elementwise_binary_op_2dim():
    tensor_0 = TensorArg(table_name="tens0", dtype=None, ndims=2)
    tensor_1 = TensorArg(table_name="tens1", dtype=None, ndims=2)
    op = TensorBinaryOp.ADD
    expected = """\
SELECT
    CAST('$node_id' AS VARCHAR(50)) AS node_id,
    tensor_0.dim0 AS dim0,
    tensor_0.dim1 AS dim1,
    tensor_0.val + tensor_1.val AS val
FROM
    tens0 AS tensor_0,
    tens1 AS tensor_1
WHERE
    tensor_0.dim0=tensor_1.dim0 AND
    tensor_0.dim1=tensor_1.dim1"""
    result = get_tensor_binary_op_template(tensor_0, tensor_1, op)
    assert result == expected


def test_vector_dot_vector_template():
    tensor_0 = TensorArg(table_name="vec0", dtype=None, ndims=1)
    tensor_1 = TensorArg(table_name="vec1", dtype=None, ndims=1)
    op = TensorBinaryOp.MATMUL
    expected = """\
SELECT
    CAST('$node_id' AS VARCHAR(50)) AS node_id,
    SUM(tensor_0.val * tensor_1.val) AS val
FROM
    vec0 AS tensor_0,
    vec1 AS tensor_1
WHERE
    tensor_0.dim0=tensor_1.dim0"""
    result = get_tensor_binary_op_template(tensor_0, tensor_1, op)
    assert result == expected


def test_matrix_dot_matrix_template():
    tensor_0 = TensorArg(table_name="mat0", dtype=None, ndims=2)
    tensor_1 = TensorArg(table_name="mat1", dtype=None, ndims=2)
    op = TensorBinaryOp.MATMUL
    expected = """\
SELECT
    CAST('$node_id' AS VARCHAR(50)) AS node_id,
    tensor_0.dim0 AS dim0,
    tensor_1.dim1 AS dim1,
    SUM(tensor_0.val * tensor_1.val) AS val
FROM
    mat0 AS tensor_0,
    mat1 AS tensor_1
WHERE
    tensor_0.dim1=tensor_1.dim0
GROUP BY
    tensor_0.dim0,
    tensor_1.dim1
ORDER BY
    dim0,
    dim1"""
    result = get_tensor_binary_op_template(tensor_0, tensor_1, op)
    assert result == expected


def test_matrix_dot_vector_template():
    tensor_0 = TensorArg(table_name="mat0", dtype=None, ndims=2)
    tensor_1 = TensorArg(table_name="vec1", dtype=None, ndims=1)
    op = TensorBinaryOp.MATMUL
    expected = """\
SELECT
    CAST('$node_id' AS VARCHAR(50)) AS node_id,
    tensor_0.dim0 AS dim0,
    SUM(tensor_0.val * tensor_1.val) AS val
FROM
    mat0 AS tensor_0,
    vec1 AS tensor_1
WHERE
    tensor_0.dim1=tensor_1.dim0
GROUP BY
    tensor_0.dim0
ORDER BY
    dim0"""
    result = get_tensor_binary_op_template(tensor_0, tensor_1, op)
    assert result == expected


def test_vector_dot_matrix_template():
    tensor_0 = TensorArg(table_name="vec0", dtype=None, ndims=1)
    tensor_1 = TensorArg(table_name="mat1", dtype=None, ndims=2)
    op = TensorBinaryOp.MATMUL
    expected = """\
SELECT
    CAST('$node_id' AS VARCHAR(50)) AS node_id,
    tensor_1.dim1 AS dim0,
    SUM(tensor_0.val * tensor_1.val) AS val
FROM
    vec0 AS tensor_0,
    mat1 AS tensor_1
WHERE
    tensor_0.dim0=tensor_1.dim0
GROUP BY
    tensor_1.dim1
ORDER BY
    dim0"""
    result = get_tensor_binary_op_template(tensor_0, tensor_1, op)
    assert result == expected


def test_sql_matrix_transpose():
    tens = TensorArg(table_name="tens0", dtype=None, ndims=2)
    expected = """\
SELECT
    CAST('$node_id' AS VARCHAR(50)) AS node_id,
    tensor_0.dim1 AS dim0,
    tensor_0.dim0 AS dim1,
    tensor_0.val AS val
FROM
    tens0 AS tensor_0"""
    result = get_matrix_transpose_template(tens)
    assert result == expected


def test_tensor_number_binary_op_1dim():
    operand_0 = TensorArg(table_name="tens0", dtype=None, ndims=1)
    operand_1 = LiteralArg(value=1)
    op = TensorBinaryOp.ADD
    expected = """\
SELECT
    CAST('$node_id' AS VARCHAR(50)) AS node_id,
    tensor_0.dim0 AS dim0,
    tensor_0.val + 1 AS val
FROM
    tens0 AS tensor_0"""
    result = get_tensor_binary_op_template(operand_0, operand_1, op)
    assert result == expected


def test_number_tensor_binary_op_1dim():
    operand_0 = LiteralArg(value=1)
    operand_1 = TensorArg(table_name="tens1", dtype=None, ndims=1)
    op = TensorBinaryOp.SUB
    expected = """\
SELECT
    CAST('$node_id' AS VARCHAR(50)) AS node_id,
    tensor_0.dim0 AS dim0,
    1 - tensor_0.val AS val
FROM
    tens1 AS tensor_0"""
    result = get_tensor_binary_op_template(operand_0, operand_1, op)
    assert result == expected
