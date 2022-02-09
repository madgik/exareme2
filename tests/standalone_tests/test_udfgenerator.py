# type: ignore
from string import Template
from typing import TypeVar
import pickle
import json

import pytest

from mipengine.datatypes import DType
from mipengine.node_tasks_DTOs import (
    ColumnInfo,
    TableInfo,
    TableSchema,
    TableType,
)
from mipengine.udfgen import secure_transfer
from mipengine.udfgen.udfgenerator import (
    Column,
    IOType,
    LiteralArg,
    MergeTensorType,
    RelationArg,
    RelationType,
    ScalarFunction,
    Select,
    StateArg,
    Table,
    TableFunction,
    TensorArg,
    TensorBinaryOp,
    TransferArg,
    UDFBadCall,
    UDFBadDefinition,
    convert_udfgenargs_to_udfargs,
    copy_types_from_udfargs,
    generate_udf_queries,
    get_funcparts_from_udf_registry,
    get_matrix_transpose_template,
    get_tensor_binary_op_template,
    get_udf_templates_using_udfregistry,
    literal,
    map_unknown_to_known_typeparams,
    mapping_inverse,
    mappings_coincide,
    merge_mappings_consistently,
    merge_transfer,
    recursive_repr,
    relation,
    scalar,
    state,
    tensor,
    transfer,
    udf,
    verify_declared_typeparams_match_passed_type,
    udf_logger,
)
from mipengine.udfgen.udfgen_DTOs import SMPCTablesInfo
from mipengine.udfgen.udfgen_DTOs import SMPCUDFGenResult
from mipengine.udfgen.udfgen_DTOs import TableUDFGenResult


REQUEST_ID = "test_udfgenerator"


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
        assert get_funcparts_from_udf_registry(funcname, some_udfregistry) is not None

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

        assert "Input types of func are not subclasses of InputType" in str(exc)

    def test_validate_func_as_udf_invalid_output_type(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(x=tensor(int, 1), return_type=bool)
            def f_invalid_sig(x):
                x = 1
                return x

        assert "Output type of func is not subclass of OutputType" in str(exc)

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

        assert "The parameters: y were not provided in the func definition." in str(exc)

    def test_validate_func_as_udf_undeclared_parameter_names(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(y=tensor(int, 1), return_type=scalar(int))
            def f(y, x):
                return x

        assert "The parameters: x were not defined in the decorator." in str(exc)

    def test_validate_func_as_udf_no_return_type(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(x=tensor(int, 1))
            def f(x):
                return x

        assert "No return_type defined." in str(exc)

    def test_validate_func_as_valid_udf_with_state_and_transfer_input(self):
        @udf(
            x=tensor(int, 1),
            y=state(),
            z=transfer(),
            return_type=scalar(int),
        )
        def f(x, y, z):
            return x

        assert udf.registry != {}

    def test_validate_func_as_valid_udf_with_transfer_output(self):
        @udf(x=tensor(int, 1), return_type=transfer())
        def f(x):
            y = {"num": 1}
            return y

        assert udf.registry != {}

    def test_validate_func_as_valid_udf_with_state_output(self):
        @udf(
            x=state(),
            y=transfer(),
            return_type=transfer(),
        )
        def f(x, y):
            y = {"num": 1}
            return y

        assert udf.registry != {}

    def test_validate_func_as_valid_udf_with_merge_transfer_input(self):
        @udf(
            x=state(),
            y=merge_transfer(),
            return_type=state(),
        )
        def f(x, y):
            y = {"num": 1}
            return y

        assert udf.registry != {}

    def test_validate_func_as_valid_udf_with_local_step_logic_and_state_main_return(
        self,
    ):
        @udf(x=state(), y=transfer(), return_type=[state(), transfer()])
        def f(x, y):
            r1 = {"num1": 1}
            r2 = {"num2": 2}
            return r1, r2

        assert udf.registry != {}

    def test_validate_func_as_valid_udf_with_local_step_logic_and_transfer_main_return(
        self,
    ):
        @udf(x=state(), y=transfer(), return_type=[transfer(), state()])
        def f(x, y):
            r1 = {"num1": 1}
            r2 = {"num2": 2}
            return r1, r2

        assert udf.registry != {}

    def test_validate_func_as_valid_udf_with_global_step_logic(self):
        @udf(x=state(), y=merge_transfer(), return_type=[state(), transfer()])
        def f(x, y):
            r1 = {"num1": 1}
            r2 = {"num2": 2}
            return r1, r2

        assert udf.registry != {}

    def test_validate_func_as_invalid_udf_with_tensor_as_sec_return(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(
                x=state(),
                y=merge_transfer(),
                return_type=[state(), tensor(DType.INT, 2)],
            )
            def f(x, y):
                r1 = {"num1": 1}
                r2 = {"num2": 2}
                return r1, r2

        assert (
            "The secondary output types of func are not subclasses of LoopbackOutputType:"
            in str(exc)
        )

    def test_validate_func_as_invalid_udf_with_relation_as_sec_return(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(
                x=state(),
                y=merge_transfer(),
                return_type=[state(), relation(schema=[])],
            )
            def f(x, y):
                r1 = {"num1": 1}
                r2 = {"num2": 2}
                return r1, r2

        assert (
            "The secondary output types of func are not subclasses of LoopbackOutputType:"
            in str(exc)
        )

    def test_validate_func_as_invalid_udf_with_scalar_as_sec_return(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(
                x=state(), y=merge_transfer(), return_type=[state(), scalar(DType.INT)]
            )
            def f(x, y):
                r1 = {"num1": 1}
                r2 = {"num2": 2}
                return r1, r2

        assert (
            "The secondary output types of func are not subclasses of LoopbackOutputType:"
            in str(exc)
        )

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

    def test_validate_func_as_valid_udf_with_secure_transfer_output(self):
        @udf(
            y=state(),
            return_type=secure_transfer(add_op=True),
        )
        def f(y):
            y = {"num": 1}
            return y

        assert udf.registry != {}

    def test_validate_func_as_valid_udf_with_secure_transfer_input(self):
        @udf(
            y=secure_transfer(add_op=True),
            return_type=transfer(),
        )
        def f(y):
            y = {"num": 1}
            return y

        assert udf.registry != {}

    def test_validate_func_as_valid_udf_with_logger_input(self):
        @udf(
            y=transfer(),
            logger=udf_logger(),
            return_type=transfer(),
        )
        def f(y, logger):
            y = {"num": 1}
            return y

        assert udf.registry != {}

    def test_validate_func_as_invalid_if_logger_is_not_the_last_input_parameter(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(
                logger=udf_logger(),
                y=transfer(),
                return_type=transfer(),
            )
            def f(logger, y):
                y = {"num": 1}
                return y

        assert "'udf_logger' must be the last input parameter" in str(exc)

    def test_validate_func_as_invalid_if_logger_exists_more_than_once(self):
        with pytest.raises(UDFBadDefinition) as exc:

            @udf(
                y=transfer(),
                logger1=udf_logger(),
                logger2=udf_logger(),
                return_type=transfer(),
            )
            def f(y, logger1, logger2):
                y = {"num": 1}
                return y

        assert "Only one 'udf_logger' parameter can exist" in str(exc)


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


def test_transfer_schema():
    to = transfer()
    assert to.schema == [("transfer", DType.JSON)]


def test_state_schema():
    to = state()
    assert to.schema == [("state", DType.BINARY)]


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
            type_=TableType.NORMAL,
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
            type_=TableType.NORMAL,
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
            type_=TableType.NORMAL,
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
            type_=TableType.NORMAL,
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


def test_convert_udfgenargs_to_transfer_udfargs():
    udfgen_posargs = [
        TableInfo(
            name="tab",
            schema_=TableSchema(
                columns=[
                    ColumnInfo(name="transfer", dtype=DType.JSON),
                ]
            ),
            type_=TableType.REMOTE,
        ),
    ]
    expected_udf_posargs = [TransferArg(table_name="tab")]
    result, _ = convert_udfgenargs_to_udfargs(udfgen_posargs, {})
    assert result == expected_udf_posargs


def test_convert_udfgenargs_to_state_udfargs():
    udfgen_posargs = [
        TableInfo(
            name="tab",
            schema_=TableSchema(
                columns=[
                    ColumnInfo(name="state", dtype=DType.BINARY),
                ]
            ),
            type_=TableType.NORMAL,
        ),
    ]
    expected_udf_posargs = [StateArg(table_name="tab")]
    result, _ = convert_udfgenargs_to_udfargs(udfgen_posargs, {})
    assert result == expected_udf_posargs


def test_convert_udfgenargs_to_state_udfargs_not_local():
    udfgen_posargs = [
        TableInfo(
            name="tab",
            schema_=TableSchema(
                columns=[
                    ColumnInfo(name="state", dtype=DType.BINARY),
                ]
            ),
            type_=TableType.REMOTE,
        ),
    ]

    with pytest.raises(UDFBadCall):
        convert_udfgenargs_to_udfargs(udfgen_posargs, {})


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
        # assert len(udfregistry) == 1
        return next(iter(udfregistry.keys()))

    @staticmethod
    def _get_udf_output_tablename_template_mapping(expected_udf_outputs):
        template_mapping = {}
        for udf_output in expected_udf_outputs:
            if isinstance(udf_output, TableUDFGenResult):
                tablename_placeholder = udf_output.tablename_placeholder
                template_mapping[tablename_placeholder] = tablename_placeholder
            elif isinstance(udf_output, SMPCUDFGenResult):
                tablename_placeholder = udf_output.template.tablename_placeholder
                template_mapping[tablename_placeholder] = tablename_placeholder
                if udf_output.add_op_values:
                    tablename_placeholder = (
                        udf_output.add_op_values.tablename_placeholder
                    )
                    template_mapping[tablename_placeholder] = tablename_placeholder
                if udf_output.min_op_values:
                    tablename_placeholder = (
                        udf_output.min_op_values.tablename_placeholder
                    )
                    template_mapping[tablename_placeholder] = tablename_placeholder
                if udf_output.max_op_values:
                    tablename_placeholder = (
                        udf_output.max_op_values.tablename_placeholder
                    )
                    template_mapping[tablename_placeholder] = tablename_placeholder
                if udf_output.union_op_values:
                    tablename_placeholder = (
                        udf_output.union_op_values.tablename_placeholder
                    )
                    template_mapping[tablename_placeholder] = tablename_placeholder
            else:
                pytest.fail(
                    f"A udf_output must be of the format TableUDFOutput or SMPCUDFOutput."
                )
        return template_mapping

    @pytest.fixture(scope="class")
    def concrete_udf_def(self, expected_udfdef, expected_udf_outputs):
        """
        Replaces the udf_name, node_id placeholders in the Template.
        If the udf has loopback tables, it also replaces their names' placeholders.
        """
        template_mapping = {
            "udf_name": "udf_test",
            "node_id": "1",
        }
        template_mapping.update(
            self._get_udf_output_tablename_template_mapping(expected_udf_outputs)
        )
        return Template(expected_udfdef).substitute(**template_mapping)

    @pytest.fixture(scope="class")
    def concrete_udf_sel(self, expected_udf_outputs, expected_udfsel):
        """
        The udf definition could contain more than one tablename placeholders.
        The expected_udf_outputs are used to replace all the necessary fields.
        Just like in the `concrete_udf_outputs` it replaces the tablename_placeholder
        in the Templates using the same tablename.
        """
        template_mapping = {
            "udf_name": "udf_test",
            "node_id": "1",
        }
        template_mapping.update(
            self._get_udf_output_tablename_template_mapping(expected_udf_outputs)
        )
        return Template(expected_udfsel).substitute(**template_mapping)

    @staticmethod
    def _concrete_table_udf_outputs(output: TableUDFGenResult):
        queries = []
        template_mapping = {output.tablename_placeholder: output.tablename_placeholder}
        queries.append(output.drop_query.substitute(**template_mapping))
        queries.append(output.create_query.substitute(**template_mapping))
        return queries

    @pytest.fixture(scope="class")
    def concrete_udf_outputs(self, expected_udf_outputs):
        """
        Replaces the tablename_placeholder in the UDFOutput(s) using the same tablename.
        """
        queries = []
        for udf_output in expected_udf_outputs:
            if isinstance(udf_output, TableUDFGenResult):
                queries.extend(self._concrete_table_udf_outputs(udf_output))
            elif isinstance(udf_output, SMPCUDFGenResult):
                queries.extend(self._concrete_table_udf_outputs(udf_output.template))
                if udf_output.add_op_values:
                    queries.extend(
                        self._concrete_table_udf_outputs(udf_output.add_op_values)
                    )
                if udf_output.min_op_values:
                    queries.extend(
                        self._concrete_table_udf_outputs(udf_output.min_op_values)
                    )
                if udf_output.max_op_values:
                    queries.extend(
                        self._concrete_table_udf_outputs(udf_output.max_op_values)
                    )
                if udf_output.union_op_values:
                    queries.extend(
                        self._concrete_table_udf_outputs(udf_output.union_op_values)
                    )
            else:
                pytest.fail(
                    f"A udf_output must be of the format TableUDFOutput or SMPCUDFOutput."
                )

        return "\n".join(queries)

    @pytest.fixture(scope="function")
    def create_transfer_table(self, globalnode_db_cursor):
        globalnode_db_cursor.execute(
            "CREATE TABLE test_transfer_table(node_id VARCHAR(500), transfer CLOB)"
        )
        globalnode_db_cursor.execute(
            "INSERT INTO test_transfer_table(node_id, transfer) VALUES(1, '{\"num\":5}')"
        )

    @pytest.fixture(scope="function")
    def create_state_table(self, globalnode_db_cursor):
        state = pickle.dumps({"num": 5}).hex()
        globalnode_db_cursor.execute(
            "CREATE TABLE test_state_table(node_id VARCHAR(500), state BLOB)"
        )
        insert_state = (
            f"INSERT INTO test_state_table(node_id, state) VALUES(1, '{state}')"
        )
        globalnode_db_cursor.execute(insert_state)

    @pytest.fixture(scope="function")
    def create_merge_transfer_table(self, globalnode_db_cursor):
        globalnode_db_cursor.execute(
            "CREATE TABLE test_merge_transfer_table(node_id VARCHAR(500), transfer CLOB)"
        )
        globalnode_db_cursor.execute(
            "INSERT INTO test_merge_transfer_table(node_id, transfer) VALUES(1, '{\"num\":5}')"
        )
        globalnode_db_cursor.execute(
            "INSERT INTO test_merge_transfer_table(node_id, transfer) VALUES(2, '{\"num\":10}')"
        )

    @pytest.fixture(scope="function")
    def create_secure_transfer_table(self, globalnode_db_cursor):
        globalnode_db_cursor.execute(
            "CREATE TABLE test_secure_transfer_table(node_id VARCHAR(500), secure_transfer CLOB)"
        )
        globalnode_db_cursor.execute(
            'INSERT INTO test_secure_transfer_table(node_id, secure_transfer) VALUES(1, \'{"sum": {"data": 1, "type": "int", "operation": "addition"}}\')'
        )
        globalnode_db_cursor.execute(
            'INSERT INTO test_secure_transfer_table(node_id, secure_transfer) VALUES(2, \'{"sum": {"data": 10, "type": "int", "operation": "addition"}}\')'
        )
        globalnode_db_cursor.execute(
            'INSERT INTO test_secure_transfer_table(node_id, secure_transfer) VALUES(3, \'{"sum": {"data": 100, "type": "int", "operation": "addition"}}\')'
        )

    @pytest.fixture(scope="function")
    def create_smpc_template_table(self, globalnode_db_cursor):
        globalnode_db_cursor.execute(
            "CREATE TABLE test_smpc_template_table(node_id VARCHAR(500), secure_transfer CLOB)"
        )
        globalnode_db_cursor.execute(
            'INSERT INTO test_smpc_template_table(node_id, secure_transfer) VALUES(1, \'{"sum": {"data": [0,1,2], "type": "int", "operation": "addition"}}\')'
        )

    @pytest.fixture(scope="function")
    def create_smpc_add_op_values_table(self, globalnode_db_cursor):
        globalnode_db_cursor.execute(
            "CREATE TABLE test_smpc_add_op_values_table(node_id VARCHAR(500), secure_transfer CLOB)"
        )
        globalnode_db_cursor.execute(
            "INSERT INTO test_smpc_add_op_values_table(node_id, secure_transfer) VALUES(1, '[100,200,300]')"
        )

    # TODO Should become more dynamic in the future.
    # It should receive a TableInfo object as input and maybe data as well.
    @pytest.fixture(scope="function")
    def create_tensor_table(self, globalnode_db_cursor):
        globalnode_db_cursor.execute(
            "CREATE TABLE tensor_in_db(node_id VARCHAR(500), dim0 INT, dim1 INT, val INT)"
        )
        globalnode_db_cursor.execute(
            "INSERT INTO tensor_in_db(node_id, dim0, dim1, val) VALUES('1', 0, 0, 3)"
        )
        globalnode_db_cursor.execute(
            "INSERT INTO tensor_in_db(node_id, dim0, dim1, val) VALUES('1', 0, 1, 4)"
        )
        globalnode_db_cursor.execute(
            "INSERT INTO tensor_in_db(node_id, dim0, dim1, val) VALUES('1', 0, 2, 7)"
        )


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
            _, _ = get_udf_templates_using_udfregistry(
                request_id=REQUEST_ID,
                funcname=funcname,
                posargs=posargs,
                keywordargs=keywordargs,
                udfregistry=udfregistry,
                smpc_used=False,
            )
        assert "UDF argument names do not match UDF parameter names" in str(exc)


class TestUDFGen_LoggerArgument_provided_in_pos_args(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            x=tensor(dtype=int, ndims=1),
            logger=udf_logger(),
            return_type=scalar(int),
        )
        def f(x, logger):
            return x

        return udf.registry

    def test_get_udf_templates(self, udfregistry, funcname):
        posargs = [TensorArg("table_name", dtype=int, ndims=1), LiteralArg(1)]
        with pytest.raises(UDFBadCall) as exc:
            _, _ = get_udf_templates_using_udfregistry(
                request_id=REQUEST_ID,
                funcname=funcname,
                posargs=posargs,
                keywordargs={},
                udfregistry=udfregistry,
                smpc_used=False,
            )
        assert "No argument should be provided for 'UDFLoggerType' parameter" in str(
            exc
        )


class TestUDFGen_LoggerArgument_provided_in_kw_args(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            x=tensor(dtype=int, ndims=1),
            logger=udf_logger(),
            return_type=scalar(int),
        )
        def f(x, logger):
            return x

        return udf.registry

    def test_get_udf_templates(self, udfregistry, funcname):
        posargs = [TensorArg("table_name", dtype=int, ndims=1)]
        keywordargs = {"logger": LiteralArg(1)}
        with pytest.raises(UDFBadCall) as exc:
            _, _ = get_udf_templates_using_udfregistry(
                request_id=REQUEST_ID,
                funcname=funcname,
                posargs=posargs,
                keywordargs=keywordargs,
                udfregistry=udfregistry,
                smpc_used=False,
            )
        assert "No argument should be provided for 'UDFLoggerType' parameter" in str(
            exc
        )


class TestUDFGen_InvalidUDFArgs_TransferTableInStateArgument(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            transfers=state(),
            state=state(),
            return_type=transfer(),
        )
        def f(transfers, state):
            result = {"num": sum}
            return result

        return udf.registry

    def test_get_udf_templates(self, udfregistry, funcname):
        posargs = [
            TableInfo(
                name="test_table_3",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="transfer", dtype=DType.JSON),
                    ]
                ),
                type_=TableType.REMOTE,
            ),
            TableInfo(
                name="test_table_5",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="state", dtype=DType.BINARY),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
        ]
        with pytest.raises(UDFBadCall) as exc:
            _, _ = generate_udf_queries(
                request_id=REQUEST_ID,
                func_name=funcname,
                positional_args=posargs,
                keyword_args={},
                smpc_used=False,
            )
        assert "should be of type" in str(exc)


class TestUDFGen_InvalidUDFArgs_TensorTableInTransferArgument(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            transfers=transfer(),
            state=state(),
            return_type=transfer(),
        )
        def f(transfers, state):
            result = {"num": sum}
            return result

        return udf.registry

    def test_get_udf_templates(self, udfregistry, funcname):
        posargs = [
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
                type_=TableType.NORMAL,
            ),
            TableInfo(
                name="test_table_5",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="state", dtype=DType.BINARY),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
        ]
        with pytest.raises(UDFBadCall) as exc:
            _ = generate_udf_queries(
                request_id=REQUEST_ID,
                func_name=funcname,
                positional_args=posargs,
                keyword_args={},
                smpc_used=True,
            )
        assert "should be of type" in str(exc)


class TestUDFGen_Invalid_SMPCUDFInput_To_Transfer_Type(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            transfer=transfer(),
            return_type=transfer(),
        )
        def f(transfer):
            result = {"num": sum}
            return result

        return udf.registry

    def test_get_udf_templates(self, udfregistry, funcname):
        posargs = [
            SMPCTablesInfo(
                template=TableInfo(
                    name="test_smpc_template_table",
                    schema_=TableSchema(
                        columns=[
                            ColumnInfo(name="secure_transfer", dtype=DType.JSON),
                        ]
                    ),
                    type_=TableType.NORMAL,
                ),
                add_op_values=TableInfo(
                    name="test_smpc_add_op_values_table",
                    schema_=TableSchema(
                        columns=[
                            ColumnInfo(name="secure_transfer", dtype=DType.JSON),
                        ]
                    ),
                    type_=TableType.NORMAL,
                ),
            )
        ]
        with pytest.raises(UDFBadCall) as exc:
            _ = generate_udf_queries(
                request_id=REQUEST_ID,
                func_name=funcname,
                positional_args=posargs,
                keyword_args={},
                smpc_used=True,
            )
        assert "should be of type" in str(exc)


class TestUDFGen_Invalid_TableInfoArgs_To_SecureTransferType(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            transfer=secure_transfer(add_op=True),
            return_type=transfer(),
        )
        def f(transfer):
            result = {"num": sum}
            return result

        return udf.registry

    def test_get_udf_templates(self, udfregistry, funcname):
        posargs = [
            TableInfo(
                name="test_table",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="secure_transfer", dtype=DType.JSON),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
        ]
        try:
            _ = generate_udf_queries(
                request_id=REQUEST_ID,
                func_name=funcname,
                positional_args=posargs,
                keyword_args={},
                smpc_used=False,
            )
        except Exception as exc:
            pytest.fail(f"An exception should not have been raised. {exc}")

        with pytest.raises(UDFBadCall) as exc:
            _ = generate_udf_queries(
                request_id=REQUEST_ID,
                func_name=funcname,
                positional_args=posargs,
                keyword_args={},
                smpc_used=True,
            )
        assert "When smpc is used SecureTransferArg should not be" in str(exc)


class TestUDFGen_Invalid_SMPCUDFInput_with_SMPC_off(TestUDFGenBase):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            transfer=secure_transfer(add_op=True),
            return_type=transfer(),
        )
        def f(transfer):
            result = {"num": sum}
            return result

        return udf.registry

    def test_get_udf_templates(self, udfregistry, funcname):
        posargs = [
            SMPCTablesInfo(
                template=TableInfo(
                    name="test_smpc_template_table",
                    schema_=TableSchema(
                        columns=[
                            ColumnInfo(name="secure_transfer", dtype=DType.JSON),
                        ]
                    ),
                    type_=TableType.NORMAL,
                ),
                add_op_values=TableInfo(
                    name="test_smpc_add_op_values_table",
                    schema_=TableSchema(
                        columns=[
                            ColumnInfo(name="secure_transfer", dtype=DType.JSON),
                        ]
                    ),
                    type_=TableType.NORMAL,
                ),
            )
        ]
        with pytest.raises(UDFBadCall) as exc:
            _ = generate_udf_queries(
                request_id=REQUEST_ID,
                func_name=funcname,
                positional_args=posargs,
                keyword_args={},
                smpc_used=False,
            )
        assert "SMPC is not used, " in str(exc)


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
            _, _ = get_udf_templates_using_udfregistry(
                request_id=REQUEST_ID,
                funcname=funcname,
                posargs=posargs,
                keywordargs=keywordargs,
                udfregistry=udfregistry,
                smpc_used=False,
            )
        err_msg, *_ = e.value.args
        assert "inconsistent mappings" in err_msg


class TestUDFGen_KW_args_on_tensor_operation:
    def test_generate_udf_queries(self):
        funcname = TensorBinaryOp.MATMUL.name
        posargs = []
        keywordargs = {"Μ": 5, "v": 7}
        with pytest.raises(UDFBadCall) as e:
            _ = generate_udf_queries(
                request_id=REQUEST_ID,
                func_name=funcname,
                positional_args=posargs,
                keyword_args=keywordargs,
                smpc_used=False,
            )
        err_msg, *_ = e.value.args
        assert "Keyword args are not supported for tensor operations." in err_msg


class _TestGenerateUDFQueries:
    """
    This class does not run as a test, it's only inherited
    when we want to add a test case for the generate_udf_queries
    """

    @pytest.fixture(scope="class")
    def udfregistry(self):
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def positional_args(self):
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def traceback(self):
        return False

    @pytest.fixture(scope="class")
    def use_smpc(self):
        return False

    @pytest.fixture(scope="class")
    def request_id(self):
        return "test_udfgenerator_base"

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfsel,
        expected_udf_outputs,
        traceback,
        use_smpc,
        request_id,
    ):
        udf_execution_queries = generate_udf_queries(
            request_id=request_id,
            func_name=funcname,
            positional_args=positional_args,
            keyword_args={},
            smpc_used=use_smpc,
            traceback=traceback,
        )
        if expected_udfdef != "":
            assert (
                udf_execution_queries.udf_definition_query.template == expected_udfdef
            )
        assert udf_execution_queries.udf_select_query.template == expected_udfsel

        for udf_output, expected_udf_output in zip(
            udf_execution_queries.udf_results, expected_udf_outputs
        ):
            assert udf_output == expected_udf_output


class TestUDFGen_TensorToTensor(TestUDFGenBase, _TestGenerateUDFQueries):
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
                type_=TableType.NORMAL,
            )
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name("x_dim0" INT,"x_dim1" INT,"x_val" INT)
RETURNS
TABLE("dim0" INT,"dim1" INT,"val" DOUBLE)
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
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
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

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"dim0" INT,"dim1" INT,"val" DOUBLE);'
                ),
            )
        ]


class TestUDFGen_TensorParameterWithCapitalLetter(
    TestUDFGenBase, _TestGenerateUDFQueries
):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        T = TypeVar("T")

        @udf(X=tensor(dtype=T, ndims=2), return_type=tensor(dtype=DType.FLOAT, ndims=2))
        def f(X):
            result = X
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
                type_=TableType.NORMAL,
            )
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name("X_dim0" INT,"X_dim1" INT,"X_val" INT)
RETURNS
TABLE("dim0" INT,"dim1" INT,"val" DOUBLE)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    X = udfio.from_tensor_table({n: _columns[n] for n in ['X_dim0', 'X_dim1', 'X_val']})
    result = X
    return udfio.as_tensor_table(numpy.array(result))
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
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

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"dim0" INT,"dim1" INT,"val" DOUBLE);'
                ),
            )
        ]

    @pytest.mark.database
    @pytest.mark.usefixtures("use_globalnode_database", "create_tensor_table")
    def test_udf_with_db(
        self,
        concrete_udf_outputs,
        concrete_udf_def,
        concrete_udf_sel,
        globalnode_db_cursor,
        create_tensor_table,
    ):
        globalnode_db_cursor.execute(concrete_udf_outputs)
        globalnode_db_cursor.execute(concrete_udf_def)
        globalnode_db_cursor.execute(concrete_udf_sel)
        output_table_values = globalnode_db_cursor.execute(
            "SELECT * FROM main_output_table_name"
        ).fetchall()
        assert output_table_values == [
            ("1", 0, 0, 3.0),
            ("1", 0, 1, 4.0),
            ("1", 0, 2, 7.0),
        ]


class TestUDFGen_RelationToTensor(TestUDFGenBase, _TestGenerateUDFQueries):
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
                type_=TableType.NORMAL,
            )
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name("r_col0" INT,"r_col1" DOUBLE,"r_col2" VARCHAR(500))
RETURNS
TABLE("dim0" INT,"dim1" INT,"val" DOUBLE)
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
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
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

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"dim0" INT,"dim1" INT,"val" DOUBLE);'
                ),
            )
        ]


class TestUDFGen_2RelationsToTensor(TestUDFGenBase, _TestGenerateUDFQueries):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        S = TypeVar("S")

        @udf(
            r1=relation(schema=S),
            r2=relation(schema=S),
            return_type=tensor(dtype=DType.FLOAT, ndims=2),
        )
        def f(r1, r2):
            result = r1
            return result

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="rel1_in_db",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="col0", dtype=DType.INT),
                        ColumnInfo(name="col1", dtype=DType.FLOAT),
                        ColumnInfo(name="col2", dtype=DType.STR),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
            TableInfo(
                name="rel2_in_db",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="col4", dtype=DType.INT),
                        ColumnInfo(name="col5", dtype=DType.FLOAT),
                        ColumnInfo(name="col6", dtype=DType.STR),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name("r1_col0" INT,"r1_col1" DOUBLE,"r1_col2" VARCHAR(500),"r2_col4" INT,"r2_col5" DOUBLE,"r2_col6" VARCHAR(500))
RETURNS
TABLE("dim0" INT,"dim1" INT,"val" DOUBLE)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    r1 = pd.DataFrame({n: _columns[n] for n in ['r1_col0', 'r1_col1', 'r1_col2']})
    r2 = pd.DataFrame({n: _columns[n] for n in ['r2_col4', 'r2_col5', 'r2_col6']})
    result = r1
    return udfio.as_tensor_table(numpy.array(result))
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    *
FROM
    $udf_name((
        SELECT
            rel1_in_db.col0,
            rel1_in_db.col1,
            rel1_in_db.col2,
            rel2_in_db.col4,
            rel2_in_db.col5,
            rel2_in_db.col6
        FROM
            rel1_in_db,
            rel2_in_db
        WHERE
            rel1_in_db.row_id=rel2_in_db.row_id
    ));"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"dim0" INT,"dim1" INT,"val" DOUBLE);'
                ),
            )
        ]


class TestUDFGen_3RelationsToTensor(TestUDFGenBase, _TestGenerateUDFQueries):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        S = TypeVar("S")

        @udf(
            r1=relation(schema=S),
            r2=relation(schema=S),
            r3=relation(schema=S),
            return_type=tensor(dtype=DType.FLOAT, ndims=2),
        )
        def f(r1, r2, r3):
            result = r1
            return result

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="rel1_in_db",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="col0", dtype=DType.INT),
                        ColumnInfo(name="col1", dtype=DType.FLOAT),
                        ColumnInfo(name="col2", dtype=DType.STR),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
            TableInfo(
                name="rel2_in_db",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="col4", dtype=DType.INT),
                        ColumnInfo(name="col5", dtype=DType.FLOAT),
                        ColumnInfo(name="col6", dtype=DType.STR),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
            TableInfo(
                name="rel3_in_db",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="col8", dtype=DType.INT),
                        ColumnInfo(name="col9", dtype=DType.FLOAT),
                        ColumnInfo(name="col10", dtype=DType.STR),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name("r1_col0" INT,"r1_col1" DOUBLE,"r1_col2" VARCHAR(500),"r2_col4" INT,"r2_col5" DOUBLE,"r2_col6" VARCHAR(500),"r3_col8" INT,"r3_col9" DOUBLE,"r3_col10" VARCHAR(500))
RETURNS
TABLE("dim0" INT,"dim1" INT,"val" DOUBLE)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    r1 = pd.DataFrame({n: _columns[n] for n in ['r1_col0', 'r1_col1', 'r1_col2']})
    r2 = pd.DataFrame({n: _columns[n] for n in ['r2_col4', 'r2_col5', 'r2_col6']})
    r3 = pd.DataFrame({n: _columns[n] for n in ['r3_col8', 'r3_col9', 'r3_col10']})
    result = r1
    return udfio.as_tensor_table(numpy.array(result))
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    *
FROM
    $udf_name((
        SELECT
            rel1_in_db.col0,
            rel1_in_db.col1,
            rel1_in_db.col2,
            rel2_in_db.col4,
            rel2_in_db.col5,
            rel2_in_db.col6,
            rel3_in_db.col8,
            rel3_in_db.col9,
            rel3_in_db.col10
        FROM
            rel1_in_db,
            rel2_in_db,
            rel3_in_db
        WHERE
            rel1_in_db.row_id=rel2_in_db.row_id AND
            rel1_in_db.row_id=rel3_in_db.row_id
    ));"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"dim0" INT,"dim1" INT,"val" DOUBLE);'
                ),
            )
        ]


class TestUDFGen_2SameRelationsToTensor(TestUDFGenBase, _TestGenerateUDFQueries):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        S = TypeVar("S")

        @udf(
            r1=relation(schema=S),
            r2=relation(schema=S),
            return_type=tensor(dtype=DType.FLOAT, ndims=2),
        )
        def f(r1, r2):
            result = r1
            return result

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="rel1_in_db",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="col0", dtype=DType.INT),
                        ColumnInfo(name="col1", dtype=DType.FLOAT),
                        ColumnInfo(name="col2", dtype=DType.STR),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
            TableInfo(
                name="rel1_in_db",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="col0", dtype=DType.INT),
                        ColumnInfo(name="col1", dtype=DType.FLOAT),
                        ColumnInfo(name="col2", dtype=DType.STR),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name("r1_col0" INT,"r1_col1" DOUBLE,"r1_col2" VARCHAR(500),"r2_col0" INT,"r2_col1" DOUBLE,"r2_col2" VARCHAR(500))
RETURNS
TABLE("dim0" INT,"dim1" INT,"val" DOUBLE)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    r1 = pd.DataFrame({n: _columns[n] for n in ['r1_col0', 'r1_col1', 'r1_col2']})
    r2 = pd.DataFrame({n: _columns[n] for n in ['r2_col0', 'r2_col1', 'r2_col2']})
    result = r1
    return udfio.as_tensor_table(numpy.array(result))
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    *
FROM
    $udf_name((
        SELECT
            rel1_in_db.col0,
            rel1_in_db.col1,
            rel1_in_db.col2,
            rel1_in_db.col0,
            rel1_in_db.col1,
            rel1_in_db.col2
        FROM
            rel1_in_db
        WHERE
            rel1_in_db.row_id=rel1_in_db.row_id
    ));"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"dim0" INT,"dim1" INT,"val" DOUBLE);'
                ),
            )
        ]


class TestUDFGen_TensorToRelation(TestUDFGenBase, _TestGenerateUDFQueries):
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
                type_=TableType.NORMAL,
            )
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name("x_dim0" INT,"x_val" INT)
RETURNS
TABLE("ci" INT,"cf" DOUBLE)
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
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    *
FROM
    $udf_name((
        SELECT
            tensor_in_db.dim0,
            tensor_in_db.val
        FROM
            tensor_in_db
    ));"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"ci" INT,"cf" DOUBLE);'
                ),
            )
        ]


class TestUDFGen_LiteralArgument(TestUDFGenBase, _TestGenerateUDFQueries):
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
                        ColumnInfo(name="row_id", dtype=DType.INT),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
            42,
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name("x_dim0" INT,"x_val" INT)
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
INSERT INTO $main_output_table_name
SELECT
    $udf_name(the_table.dim0,the_table.val)
FROM
    the_table;"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("result" INT);'
                ),
            )
        ]


class TestUDFGen_ManyLiteralArguments(TestUDFGenBase, _TestGenerateUDFQueries):
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
                        ColumnInfo(name="row_id", dtype=DType.INT),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
            42,
            24,
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name("x_dim0" INT,"x_val" INT)
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
INSERT INTO $main_output_table_name
SELECT
    $udf_name(tensor_in_db.dim0,tensor_in_db.val)
FROM
    tensor_in_db;"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("result" INT);'
                ),
            )
        ]


class TestUDFGen_NoArguments(TestUDFGenBase, _TestGenerateUDFQueries):
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
TABLE("dim0" INT,"val" INT)
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
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    *
FROM
    $udf_name();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"dim0" INT,"val" INT);'
                ),
            )
        ]


class TestUDFGen_RelationInExcludeRowId(TestUDFGenBase, _TestGenerateUDFQueries):
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
                        ColumnInfo(name="row_id", dtype=DType.INT),
                        ColumnInfo(name="c0", dtype=DType.INT),
                        ColumnInfo(name="c1", dtype=DType.FLOAT),
                        ColumnInfo(name="c2", dtype=DType.STR),
                    ]
                ),
                type_=TableType.NORMAL,
            )
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name("r_c0" INT,"r_c1" DOUBLE,"r_c2" VARCHAR(500))
RETURNS
TABLE("dim0" INT,"dim1" INT,"val" DOUBLE)
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
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
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

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"dim0" INT,"dim1" INT,"val" DOUBLE);'
                ),
            )
        ]


class TestUDFGen_UnknownReturnDimensions(TestUDFGenBase, _TestGenerateUDFQueries):
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
                        ColumnInfo(name="row_id", dtype=DType.INT),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="dim1", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
                type_=TableType.NORMAL,
            )
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name("t_dim0" INT,"t_dim1" INT,"t_val" INT)
RETURNS
TABLE("dim0" INT,"dim1" INT,"val" INT)
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
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
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

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"dim0" INT,"dim1" INT,"val" INT);'
                ),
            )
        ]


class TestUDFGen_TwoTensors1DReturnTable(TestUDFGenBase, _TestGenerateUDFQueries):
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
                        ColumnInfo(name="row_id", dtype=DType.INT),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
            TableInfo(
                name="tens1",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.INT),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name("x_dim0" INT,"x_val" INT,"y_dim0" INT,"y_val" INT)
RETURNS
TABLE("dim0" INT,"val" INT)
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
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
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

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"dim0" INT,"val" INT);'
                ),
            )
        ]


class TestUDFGen_ThreeTensors1DReturnTable(TestUDFGenBase, _TestGenerateUDFQueries):
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
                        ColumnInfo(name="row_id", dtype=DType.INT),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
            TableInfo(
                name="tens1",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.INT),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
            TableInfo(
                name="tens2",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.INT),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name("x_dim0" INT,"x_val" INT,"y_dim0" INT,"y_val" INT,"z_dim0" INT,"z_val" INT)
RETURNS
TABLE("dim0" INT,"val" INT)
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
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
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

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"dim0" INT,"val" INT);'
                ),
            )
        ]


class TestUDFGen_ThreeTensors2DReturnTable(TestUDFGenBase, _TestGenerateUDFQueries):
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
                        ColumnInfo(name="row_id", dtype=DType.INT),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="dim1", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
            TableInfo(
                name="tens1",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.INT),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="dim1", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
            TableInfo(
                name="tens2",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.INT),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="dim1", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name("x_dim0" INT,"x_dim1" INT,"x_val" INT,"y_dim0" INT,"y_dim1" INT,"y_val" INT,"z_dim0" INT,"z_dim1" INT,"z_val" INT)
RETURNS
TABLE("dim0" INT,"dim1" INT,"val" INT)
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
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
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

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"dim0" INT,"dim1" INT,"val" INT);'
                ),
            )
        ]


class TestUDFGen_TwoTensors1DReturnScalar(TestUDFGenBase, _TestGenerateUDFQueries):
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
                        ColumnInfo(name="row_id", dtype=DType.INT),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
            TableInfo(
                name="tens1",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.INT),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name("x_dim0" INT,"x_val" INT,"y_dim0" INT,"y_val" INT)
RETURNS
DOUBLE
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
INSERT INTO $main_output_table_name
SELECT
    $udf_name(tens0.dim0,tens0.val,tens1.dim0,tens1.val)
FROM
    tens0,
    tens1
WHERE
    tens0.dim0=tens1.dim0;"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("result" DOUBLE);'
                ),
            )
        ]


class TestUDFGen_SQLTensorMultOut1D(TestUDFGenBase, _TestGenerateUDFQueries):
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
                type_=TableType.NORMAL,
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
                type_=TableType.NORMAL,
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return ""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
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

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"dim0" INT,"val" DOUBLE);'
                ),
            )
        ]


class TestUDFGen_SQLTensorMultOut2D(TestUDFGenBase, _TestGenerateUDFQueries):
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
                type_=TableType.NORMAL,
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
                type_=TableType.NORMAL,
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return ""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
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

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"dim0" INT,"dim1" INT,"val" DOUBLE);'
                ),
            )
        ]


class TestUDFGen_SQLTensorSubLiteralArg(TestUDFGenBase, _TestGenerateUDFQueries):
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
                type_=TableType.NORMAL,
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return ""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    tensor_0.dim0 AS dim0,
    1 - tensor_0.val AS val
FROM
    tensor1 AS tensor_0;"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"dim0" INT,"val" DOUBLE);'
                ),
            )
        ]


class TestUDFGen_ScalarReturn(TestUDFGenBase, _TestGenerateUDFQueries):
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
                        ColumnInfo(name="row_id", dtype=DType.INT),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="dim1", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
                type_=TableType.NORMAL,
            )
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name("x_dim0" INT,"x_dim1" INT,"x_val" INT)
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
INSERT INTO $main_output_table_name
SELECT
    $udf_name(tensor_in_db.dim0,tensor_in_db.dim1,tensor_in_db.val)
FROM
    tensor_in_db;"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("result" INT);'
                ),
            )
        ]


class TestUDFGen_MergeTensor(TestUDFGenBase, _TestGenerateUDFQueries):
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
                        ColumnInfo(name="row_id", dtype=DType.INT),
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
                type_=TableType.NORMAL,
            )
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name("xs_node_id" VARCHAR(500),"xs_dim0" INT,"xs_val" INT)
RETURNS
TABLE("dim0" INT,"val" INT)
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
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
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

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"dim0" INT,"val" INT);'
                ),
            )
        ]


class TestUDFGen_TracebackFlag(TestUDFGenBase, _TestGenerateUDFQueries):
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
                type_=TableType.NORMAL,
            )
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return r"""CREATE OR REPLACE FUNCTION
$udf_name("x_dim0" INT,"x_val" INT)
RETURNS
VARCHAR(500)
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
INSERT INTO $main_output_table_name
SELECT
    $udf_name(tensor_in_db.dim0,tensor_in_db.val)
FROM
    tensor_in_db;"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("result" VARCHAR(500));'
                ),
            )
        ]

    @pytest.fixture(scope="class")
    def traceback(self):
        return True


class TestUDFGen_StateReturnType(TestUDFGenBase, _TestGenerateUDFQueries):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(t=literal(), return_type=state())
        def f(t):
            result = {"num": 5}
            return result

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [5]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name()
RETURNS
TABLE("state" BLOB)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    import pickle
    t = 5
    result = {'num': 5}
    return pickle.dumps(result)
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    *
FROM
    $udf_name();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"state" BLOB);'
                ),
            )
        ]

    @pytest.mark.database
    @pytest.mark.usefixtures("use_globalnode_database")
    def test_udf_with_db(
        self,
        concrete_udf_outputs,
        concrete_udf_def,
        concrete_udf_sel,
        globalnode_db_cursor,
    ):
        globalnode_db_cursor.execute(concrete_udf_outputs)
        globalnode_db_cursor.execute(concrete_udf_def)
        globalnode_db_cursor.execute(concrete_udf_sel)
        _, state = globalnode_db_cursor.execute(
            "SELECT * FROM main_output_table_name"
        ).fetchone()
        result = pickle.loads(state)
        assert result == {"num": 5}


class TestUDFGen_StateInputandReturnType(TestUDFGenBase, _TestGenerateUDFQueries):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            t=literal(),
            prev_state=state(),
            return_type=state(),
        )
        def f(t, prev_state):
            prev_state["num"] = prev_state["num"] + t
            return prev_state

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            5,
            TableInfo(
                name="test_state_table",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="state", dtype=DType.BINARY),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name()
RETURNS
TABLE("state" BLOB)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    import pickle
    __state_str = _conn.execute("SELECT state from test_state_table;")["state"][0]
    prev_state = pickle.loads(__state_str)
    t = 5
    prev_state['num'] = prev_state['num'] + t
    return pickle.dumps(prev_state)
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    *
FROM
    $udf_name();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"state" BLOB);'
                ),
            )
        ]

    @pytest.mark.database
    @pytest.mark.usefixtures("use_globalnode_database", "create_state_table")
    def test_udf_with_db(
        self,
        concrete_udf_outputs,
        concrete_udf_def,
        concrete_udf_sel,
        globalnode_db_cursor,
    ):
        globalnode_db_cursor.execute(concrete_udf_outputs)
        globalnode_db_cursor.execute(concrete_udf_def)
        globalnode_db_cursor.execute(concrete_udf_sel)
        _, state = globalnode_db_cursor.execute(
            "SELECT * FROM main_output_table_name"
        ).fetchone()
        result = pickle.loads(state)
        assert result == {"num": 10}


class TestUDFGen_TransferReturnType(TestUDFGenBase, _TestGenerateUDFQueries):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(t=literal(), return_type=transfer())
        def f(t):
            result = {"num": t, "list_of_nums": [t, t, t]}
            return result

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [5]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name()
RETURNS
TABLE("transfer" CLOB)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    import json
    t = 5
    result = {'num': t, 'list_of_nums': [t, t, t]}
    return json.dumps(result)
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    *
FROM
    $udf_name();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"transfer" CLOB);'
                ),
            )
        ]

    @pytest.mark.database
    @pytest.mark.usefixtures("use_globalnode_database")
    def test_udf_with_db(
        self,
        concrete_udf_outputs,
        concrete_udf_def,
        concrete_udf_sel,
        globalnode_db_cursor,
    ):
        globalnode_db_cursor.execute(concrete_udf_outputs)
        globalnode_db_cursor.execute(concrete_udf_def)
        globalnode_db_cursor.execute(concrete_udf_sel)
        _, transfer = globalnode_db_cursor.execute(
            "SELECT * FROM main_output_table_name"
        ).fetchone()
        result = json.loads(transfer)
        assert result == {"num": 5, "list_of_nums": [5, 5, 5]}


class TestUDFGen_TransferInputAndReturnType(TestUDFGenBase, _TestGenerateUDFQueries):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            t=literal(),
            transfer=transfer(),
            return_type=transfer(),
        )
        def f(t, transfer):
            transfer["num"] = transfer["num"] + t
            return transfer

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            5,
            TableInfo(
                name="test_transfer_table",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="transfer", dtype=DType.JSON),
                    ]
                ),
                type_=TableType.REMOTE,
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name()
RETURNS
TABLE("transfer" CLOB)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    import json
    __transfer_str = _conn.execute("SELECT transfer from test_transfer_table;")["transfer"][0]
    transfer = json.loads(__transfer_str)
    t = 5
    transfer['num'] = transfer['num'] + t
    return json.dumps(transfer)
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    *
FROM
    $udf_name();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"transfer" CLOB);'
                ),
            )
        ]

    @pytest.mark.database
    @pytest.mark.usefixtures("use_globalnode_database", "create_transfer_table")
    def test_udf_with_db(
        self,
        concrete_udf_outputs,
        concrete_udf_def,
        concrete_udf_sel,
        globalnode_db_cursor,
    ):
        globalnode_db_cursor.execute(concrete_udf_outputs)
        globalnode_db_cursor.execute(concrete_udf_def)
        globalnode_db_cursor.execute(concrete_udf_sel)
        _, transfer = globalnode_db_cursor.execute(
            "SELECT * FROM main_output_table_name"
        ).fetchone()
        result = json.loads(transfer)
        assert result == {"num": 10}


class TestUDFGen_TransferInputAndStateReturnType(
    TestUDFGenBase, _TestGenerateUDFQueries
):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            t=literal(),
            transfer=transfer(),
            return_type=state(),
        )
        def f(t, transfer):
            transfer["num"] = transfer["num"] + t
            return transfer

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            5,
            TableInfo(
                name="test_transfer_table",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="transfer", dtype=DType.JSON),
                    ]
                ),
                type_=TableType.REMOTE,
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name()
RETURNS
TABLE("state" BLOB)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    import pickle
    import json
    __transfer_str = _conn.execute("SELECT transfer from test_transfer_table;")["transfer"][0]
    transfer = json.loads(__transfer_str)
    t = 5
    transfer['num'] = transfer['num'] + t
    return pickle.dumps(transfer)
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    *
FROM
    $udf_name();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"state" BLOB);'
                ),
            )
        ]

    @pytest.mark.database
    @pytest.mark.usefixtures("use_globalnode_database", "create_transfer_table")
    def test_udf_with_db(
        self,
        concrete_udf_outputs,
        concrete_udf_def,
        concrete_udf_sel,
        globalnode_db_cursor,
    ):
        globalnode_db_cursor.execute(concrete_udf_outputs)
        globalnode_db_cursor.execute(concrete_udf_def)
        globalnode_db_cursor.execute(concrete_udf_sel)
        _, state = globalnode_db_cursor.execute(
            "SELECT * FROM main_output_table_name"
        ).fetchone()
        result = pickle.loads(state)
        assert result == {"num": 10}


class TestUDFGen_TransferAndStateInputandStateReturnType(
    TestUDFGenBase, _TestGenerateUDFQueries
):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            t=literal(),
            transfer=transfer(),
            state=state(),
            return_type=state(),
        )
        def f(t, transfer, state):
            result = {}
            result["num"] = transfer["num"] + state["num"] + t
            return result

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            5,
            TableInfo(
                name="test_transfer_table",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="transfer", dtype=DType.JSON),
                    ]
                ),
                type_=TableType.REMOTE,
            ),
            TableInfo(
                name="test_state_table",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="state", dtype=DType.BINARY),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name()
RETURNS
TABLE("state" BLOB)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    import pickle
    import json
    __transfer_str = _conn.execute("SELECT transfer from test_transfer_table;")["transfer"][0]
    transfer = json.loads(__transfer_str)
    __state_str = _conn.execute("SELECT state from test_state_table;")["state"][0]
    state = pickle.loads(__state_str)
    t = 5
    result = {}
    result['num'] = transfer['num'] + state['num'] + t
    return pickle.dumps(result)
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    *
FROM
    $udf_name();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"state" BLOB);'
                ),
            )
        ]

    @pytest.mark.database
    @pytest.mark.usefixtures(
        "use_globalnode_database",
        "create_transfer_table",
        "create_state_table",
    )
    def test_udf_with_db(
        self,
        concrete_udf_outputs,
        concrete_udf_def,
        concrete_udf_sel,
        globalnode_db_cursor,
    ):
        globalnode_db_cursor.execute(concrete_udf_outputs)
        globalnode_db_cursor.execute(concrete_udf_def)
        globalnode_db_cursor.execute(concrete_udf_sel)
        _, state = globalnode_db_cursor.execute(
            "SELECT * FROM main_output_table_name"
        ).fetchone()
        result = pickle.loads(state)
        assert result == {"num": 15}


class TestUDFGen_MergeTransferAndStateInputandTransferReturnType(
    TestUDFGenBase, _TestGenerateUDFQueries
):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            transfers=merge_transfer(),
            state=state(),
            return_type=transfer(),
        )
        def f(transfers, state):
            sum = 0
            for t in transfers:
                sum += t["num"]
            sum += state["num"]
            result = {"num": sum}
            return result

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="test_merge_transfer_table",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="transfer", dtype=DType.JSON),
                    ]
                ),
                type_=TableType.REMOTE,
            ),
            TableInfo(
                name="test_state_table",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="state", dtype=DType.BINARY),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name()
RETURNS
TABLE("transfer" CLOB)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    import pickle
    import json
    __transfer_strs = _conn.execute("SELECT transfer from test_merge_transfer_table;")["transfer"]
    transfers = [json.loads(str) for str in __transfer_strs]
    __state_str = _conn.execute("SELECT state from test_state_table;")["state"][0]
    state = pickle.loads(__state_str)
    sum = 0
    for t in transfers:
        sum += t['num']
    sum += state['num']
    result = {'num': sum}
    return json.dumps(result)
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    *
FROM
    $udf_name();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"transfer" CLOB);'
                ),
            )
        ]

    @pytest.mark.database
    @pytest.mark.usefixtures(
        "use_globalnode_database",
        "create_merge_transfer_table",
        "create_state_table",
    )
    def test_udf_with_db(
        self,
        concrete_udf_outputs,
        concrete_udf_def,
        concrete_udf_sel,
        globalnode_db_cursor,
    ):
        globalnode_db_cursor.execute(concrete_udf_outputs)
        globalnode_db_cursor.execute(concrete_udf_def)
        globalnode_db_cursor.execute(concrete_udf_sel)
        _, transfer = globalnode_db_cursor.execute(
            "SELECT * FROM main_output_table_name"
        ).fetchone()
        result = json.loads(transfer)
        assert result == {"num": 20}


class TestUDFGen_LocalStepLogic(TestUDFGenBase, _TestGenerateUDFQueries):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            state=state(),
            transfer=transfer(),
            return_type=[state(), transfer()],
        )
        def f(state, transfer):
            result1 = {"num": transfer["num"] + state["num"]}
            result2 = {"num": transfer["num"] * state["num"]}
            return result1, result2

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="test_state_table",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="state", dtype=DType.BINARY),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
            TableInfo(
                name="test_transfer_table",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="transfer", dtype=DType.JSON),
                    ]
                ),
                type_=TableType.REMOTE,
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name()
RETURNS
TABLE("state" BLOB)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    import pickle
    import json
    __state_str = _conn.execute("SELECT state from test_state_table;")["state"][0]
    state = pickle.loads(__state_str)
    __transfer_str = _conn.execute("SELECT transfer from test_transfer_table;")["transfer"][0]
    transfer = json.loads(__transfer_str)
    result1 = {'num': transfer['num'] + state['num']}
    result2 = {'num': transfer['num'] * state['num']}
    _conn.execute(f"INSERT INTO $loopback_table_name_0 VALUES ('$node_id', '{json.dumps(result2)}');")
    return pickle.dumps(result1)
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    *
FROM
    $udf_name();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"state" BLOB);'
                ),
            ),
            TableUDFGenResult(
                tablename_placeholder="loopback_table_name_0",
                drop_query=Template("DROP TABLE IF EXISTS $loopback_table_name_0;"),
                create_query=Template(
                    'CREATE TABLE $loopback_table_name_0("node_id" VARCHAR(500),"transfer" CLOB);'
                ),
            ),
        ]

    @pytest.mark.database
    @pytest.mark.usefixtures(
        "use_globalnode_database",
        "create_transfer_table",
        "create_state_table",
    )
    def test_udf_with_db(
        self,
        concrete_udf_outputs,
        concrete_udf_def,
        concrete_udf_sel,
        globalnode_db_cursor,
    ):
        globalnode_db_cursor.execute(concrete_udf_outputs)
        globalnode_db_cursor.execute(concrete_udf_def)
        globalnode_db_cursor.execute(concrete_udf_sel)
        (state_,) = globalnode_db_cursor.execute(
            "SELECT state FROM main_output_table_name"
        ).fetchone()
        result1 = pickle.loads(state_)
        assert result1 == {"num": 10}
        (transfer_,) = globalnode_db_cursor.execute(
            "SELECT transfer FROM loopback_table_name_0"
        ).fetchone()
        result2 = json.loads(transfer_)
        assert result2 == {"num": 25}


class TestUDFGen_LocalStepLogic_Transfer_first_input_and_output(
    TestUDFGenBase, _TestGenerateUDFQueries
):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            transfer=transfer(),
            state=state(),
            return_type=[transfer(), state()],
        )
        def f(transfer, state):
            result1 = {"num": transfer["num"] + state["num"]}
            result2 = {"num": transfer["num"] * state["num"]}
            return result1, result2

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="test_transfer_table",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="transfer", dtype=DType.JSON),
                    ]
                ),
                type_=TableType.REMOTE,
            ),
            TableInfo(
                name="test_state_table",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="state", dtype=DType.BINARY),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name()
RETURNS
TABLE("transfer" CLOB)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    import pickle
    import json
    __transfer_str = _conn.execute("SELECT transfer from test_transfer_table;")["transfer"][0]
    transfer = json.loads(__transfer_str)
    __state_str = _conn.execute("SELECT state from test_state_table;")["state"][0]
    state = pickle.loads(__state_str)
    result1 = {'num': transfer['num'] + state['num']}
    result2 = {'num': transfer['num'] * state['num']}
    _conn.execute(f"INSERT INTO $loopback_table_name_0 VALUES ('$node_id', '{pickle.dumps(result2).hex()}');")
    return json.dumps(result1)
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    *
FROM
    $udf_name();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"transfer" CLOB);'
                ),
            ),
            TableUDFGenResult(
                tablename_placeholder="loopback_table_name_0",
                drop_query=Template("DROP TABLE IF EXISTS $loopback_table_name_0;"),
                create_query=Template(
                    'CREATE TABLE $loopback_table_name_0("node_id" VARCHAR(500),"state" BLOB);'
                ),
            ),
        ]

    @pytest.mark.database
    @pytest.mark.usefixtures(
        "use_globalnode_database",
        "create_transfer_table",
        "create_state_table",
    )
    def test_udf_with_globalnode_db_cursor(
        self,
        concrete_udf_outputs,
        concrete_udf_def,
        concrete_udf_sel,
        globalnode_db_cursor,
    ):
        globalnode_db_cursor.execute(concrete_udf_outputs)
        globalnode_db_cursor.execute(concrete_udf_def)
        globalnode_db_cursor.execute(concrete_udf_sel)
        (transfer_,) = globalnode_db_cursor.execute(
            "SELECT transfer FROM main_output_table_name"
        ).fetchone()
        result1 = json.loads(transfer_)
        assert result1 == {"num": 10}
        (state_,) = globalnode_db_cursor.execute(
            "SELECT state FROM loopback_table_name_0"
        ).fetchone()
        result2 = pickle.loads(state_)
        assert result2 == {"num": 25}


class TestUDFGen_GlobalStepLogic(TestUDFGenBase, _TestGenerateUDFQueries):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            state=state(),
            transfers=merge_transfer(),
            return_type=[state(), transfer()],
        )
        def f(state, transfers):
            sum_transfers = 0
            for transfer in transfers:
                sum_transfers += transfer["num"]
            result1 = {"num": sum_transfers + state["num"]}
            result2 = {"num": sum_transfers * state["num"]}
            return result1, result2

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="test_state_table",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="state", dtype=DType.BINARY),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
            TableInfo(
                name="test_merge_transfer_table",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="transfer", dtype=DType.JSON),
                    ]
                ),
                type_=TableType.REMOTE,
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name()
RETURNS
TABLE("state" BLOB)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    import pickle
    import json
    __state_str = _conn.execute("SELECT state from test_state_table;")["state"][0]
    state = pickle.loads(__state_str)
    __transfer_strs = _conn.execute("SELECT transfer from test_merge_transfer_table;")["transfer"]
    transfers = [json.loads(str) for str in __transfer_strs]
    sum_transfers = 0
    for transfer in transfers:
        sum_transfers += transfer['num']
    result1 = {'num': sum_transfers + state['num']}
    result2 = {'num': sum_transfers * state['num']}
    _conn.execute(f"INSERT INTO $loopback_table_name_0 VALUES ('$node_id', '{json.dumps(result2)}');")
    return pickle.dumps(result1)
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    *
FROM
    $udf_name();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"state" BLOB);'
                ),
            ),
            TableUDFGenResult(
                tablename_placeholder="loopback_table_name_0",
                drop_query=Template("DROP TABLE IF EXISTS $loopback_table_name_0;"),
                create_query=Template(
                    'CREATE TABLE $loopback_table_name_0("node_id" VARCHAR(500),"transfer" CLOB);'
                ),
            ),
        ]

    @pytest.mark.database
    @pytest.mark.usefixtures(
        "use_globalnode_database",
        "create_merge_transfer_table",
        "create_state_table",
    )
    def test_udf_with_db(
        self,
        concrete_udf_outputs,
        concrete_udf_def,
        concrete_udf_sel,
        globalnode_db_cursor,
    ):
        globalnode_db_cursor.execute(concrete_udf_outputs)
        globalnode_db_cursor.execute(concrete_udf_def)
        globalnode_db_cursor.execute(concrete_udf_sel)
        (state_,) = globalnode_db_cursor.execute(
            "SELECT state FROM main_output_table_name"
        ).fetchone()
        result1 = pickle.loads(state_)
        assert result1 == {"num": 20}
        (transfer_,) = globalnode_db_cursor.execute(
            "SELECT transfer FROM loopback_table_name_0"
        ).fetchone()
        result2 = json.loads(transfer_)
        assert result2 == {"num": 75}


class TestUDFGen_SecureTransferOutput_with_SMPC_off(
    TestUDFGenBase, _TestGenerateUDFQueries
):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            state=state(),
            return_type=secure_transfer(add_op=True),
        )
        def f(state):
            result = {
                "sum": {"data": state["num"], "type": "int", "operation": "addition"}
            }
            return result

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="test_state_table",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="state", dtype=DType.BINARY),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name()
RETURNS
TABLE("secure_transfer" CLOB)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    import pickle
    import json
    __state_str = _conn.execute("SELECT state from test_state_table;")["state"][0]
    state = pickle.loads(__state_str)
    result = {'sum': {'data': state['num'], 'type': 'int', 'operation': 'addition'}
        }
    return json.dumps(result)
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    *
FROM
    $udf_name();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"secure_transfer" CLOB);'
                ),
            ),
        ]

    @pytest.mark.database
    @pytest.mark.usefixtures(
        "use_globalnode_database",
        "create_state_table",
    )
    def test_udf_with_db(
        self,
        concrete_udf_outputs,
        concrete_udf_def,
        concrete_udf_sel,
        globalnode_db_cursor,
    ):
        globalnode_db_cursor.execute(concrete_udf_outputs)
        globalnode_db_cursor.execute(concrete_udf_def)
        globalnode_db_cursor.execute(concrete_udf_sel)
        secure_transfer_, *_ = globalnode_db_cursor.execute(
            "SELECT secure_transfer FROM main_output_table_name"
        ).fetchone()
        result = json.loads(secure_transfer_)
        assert result == {"sum": {"data": 5, "type": "int", "operation": "addition"}}


class TestUDFGen_SecureTransferOutput_with_SMPC_on(
    TestUDFGenBase, _TestGenerateUDFQueries
):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            state=state(),
            return_type=secure_transfer(add_op=True),
        )
        def f(state):
            result = {
                "sum": {"data": state["num"], "type": "int", "operation": "addition"}
            }
            return result

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="test_state_table",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="state", dtype=DType.BINARY),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name()
RETURNS
TABLE("secure_transfer" CLOB)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    import pickle
    import json
    __state_str = _conn.execute("SELECT state from test_state_table;")["state"][0]
    state = pickle.loads(__state_str)
    result = {'sum': {'data': state['num'], 'type': 'int', 'operation': 'addition'}
        }
    template, add_op, min_op, max_op, union_op = udfio.split_secure_transfer_dict(result)
    _conn.execute(f"INSERT INTO $main_output_table_name_add_op VALUES ('$node_id', '{json.dumps(add_op)}');")
    return json.dumps(template)
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    *
FROM
    $udf_name();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            SMPCUDFGenResult(
                template=TableUDFGenResult(
                    tablename_placeholder="main_output_table_name",
                    drop_query=Template(
                        "DROP TABLE IF EXISTS $main_output_table_name;"
                    ),
                    create_query=Template(
                        'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"secure_transfer" CLOB);'
                    ),
                ),
                add_op_values=TableUDFGenResult(
                    tablename_placeholder="main_output_table_name_add_op",
                    drop_query=Template(
                        "DROP TABLE IF EXISTS $main_output_table_name_add_op;"
                    ),
                    create_query=Template(
                        'CREATE TABLE $main_output_table_name_add_op("node_id" VARCHAR(500),"secure_transfer" CLOB);'
                    ),
                ),
            )
        ]

    @pytest.fixture(scope="class")
    def use_smpc(self):
        return True

    @pytest.mark.database
    @pytest.mark.usefixtures(
        "use_globalnode_database",
        "create_state_table",
    )
    def test_udf_with_db(
        self,
        concrete_udf_outputs,
        concrete_udf_def,
        concrete_udf_sel,
        globalnode_db_cursor,
    ):
        globalnode_db_cursor.execute(concrete_udf_outputs)
        globalnode_db_cursor.execute(concrete_udf_def)
        globalnode_db_cursor.execute(concrete_udf_sel)
        template_str, *_ = globalnode_db_cursor.execute(
            "SELECT secure_transfer FROM main_output_table_name"
        ).fetchone()
        template = json.loads(template_str)
        assert template == {"sum": {"data": 0, "type": "int", "operation": "addition"}}

        add_op_values_str, *_ = globalnode_db_cursor.execute(
            "SELECT secure_transfer FROM main_output_table_name_add_op"
        ).fetchone()
        add_op_values = json.loads(add_op_values_str)
        assert add_op_values == [5]


class TestUDFGen_SecureTransferOutputAs2ndOutput_with_SMPC_off(
    TestUDFGenBase, _TestGenerateUDFQueries
):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            state=state(),
            return_type=[state(), secure_transfer(add_op=True)],
        )
        def f(state):
            result = {
                "sum": {"data": state["num"], "type": "int", "operation": "addition"}
            }
            return state, result

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="test_state_table",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="state", dtype=DType.BINARY),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name()
RETURNS
TABLE("state" BLOB)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    import pickle
    import json
    __state_str = _conn.execute("SELECT state from test_state_table;")["state"][0]
    state = pickle.loads(__state_str)
    result = {'sum': {'data': state['num'], 'type': 'int', 'operation': 'addition'}
        }
    _conn.execute(f"INSERT INTO $loopback_table_name_0 VALUES ('$node_id', '{json.dumps(result)}');")
    return pickle.dumps(state)
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    *
FROM
    $udf_name();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"state" BLOB);'
                ),
            ),
            TableUDFGenResult(
                tablename_placeholder="loopback_table_name_0",
                drop_query=Template("DROP TABLE IF EXISTS $loopback_table_name_0;"),
                create_query=Template(
                    'CREATE TABLE $loopback_table_name_0("node_id" VARCHAR(500),"secure_transfer" CLOB);'
                ),
            ),
        ]

    @pytest.mark.database
    @pytest.mark.usefixtures(
        "use_globalnode_database",
        "create_state_table",
    )
    def test_udf_with_db(
        self,
        concrete_udf_outputs,
        concrete_udf_def,
        concrete_udf_sel,
        globalnode_db_cursor,
    ):
        globalnode_db_cursor.execute(concrete_udf_outputs)
        globalnode_db_cursor.execute(concrete_udf_def)
        globalnode_db_cursor.execute(concrete_udf_sel)

        secure_transfer_, *_ = globalnode_db_cursor.execute(
            "SELECT secure_transfer FROM loopback_table_name_0"
        ).fetchone()
        result = json.loads(secure_transfer_)
        assert result == {"sum": {"data": 5, "type": "int", "operation": "addition"}}


class TestUDFGen_SecureTransferOutputAs2ndOutput_with_SMPC_on(
    TestUDFGenBase, _TestGenerateUDFQueries
):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            state=state(),
            return_type=[state(), secure_transfer(add_op=True)],
        )
        def f(state):
            result = {
                "sum": {"data": state["num"], "type": "int", "operation": "addition"}
            }
            return state, result

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="test_state_table",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="state", dtype=DType.BINARY),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name()
RETURNS
TABLE("state" BLOB)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    import pickle
    import json
    __state_str = _conn.execute("SELECT state from test_state_table;")["state"][0]
    state = pickle.loads(__state_str)
    result = {'sum': {'data': state['num'], 'type': 'int', 'operation': 'addition'}
        }
    template, add_op, min_op, max_op, union_op = udfio.split_secure_transfer_dict(result)
    _conn.execute(f"INSERT INTO $loopback_table_name_0 VALUES ('$node_id', '{json.dumps(template)}');")
    _conn.execute(f"INSERT INTO $loopback_table_name_0_add_op VALUES ('$node_id', '{json.dumps(add_op)}');")
    return pickle.dumps(state)
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    *
FROM
    $udf_name();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"state" BLOB);'
                ),
            ),
            SMPCUDFGenResult(
                template=TableUDFGenResult(
                    tablename_placeholder="loopback_table_name_0",
                    drop_query=Template("DROP TABLE IF EXISTS $loopback_table_name_0;"),
                    create_query=Template(
                        'CREATE TABLE $loopback_table_name_0("node_id" VARCHAR(500),"secure_transfer" CLOB);'
                    ),
                ),
                add_op_values=TableUDFGenResult(
                    tablename_placeholder="loopback_table_name_0_add_op",
                    drop_query=Template(
                        "DROP TABLE IF EXISTS $loopback_table_name_0_add_op;"
                    ),
                    create_query=Template(
                        'CREATE TABLE $loopback_table_name_0_add_op("node_id" VARCHAR(500),"secure_transfer" CLOB);'
                    ),
                ),
            ),
        ]

    @pytest.fixture(scope="class")
    def use_smpc(self):
        return True

    @pytest.mark.database
    @pytest.mark.usefixtures(
        "use_globalnode_database",
        "create_state_table",
    )
    def test_udf_with_db(
        self,
        concrete_udf_outputs,
        concrete_udf_def,
        concrete_udf_sel,
        globalnode_db_cursor,
    ):
        globalnode_db_cursor.execute(concrete_udf_outputs)
        globalnode_db_cursor.execute(concrete_udf_def)
        globalnode_db_cursor.execute(concrete_udf_sel)
        template_str, *_ = globalnode_db_cursor.execute(
            "SELECT secure_transfer FROM loopback_table_name_0"
        ).fetchone()
        template = json.loads(template_str)
        assert template == {"sum": {"data": 0, "type": "int", "operation": "addition"}}

        add_op_values_str, *_ = globalnode_db_cursor.execute(
            "SELECT secure_transfer FROM loopback_table_name_0_add_op"
        ).fetchone()
        add_op_values = json.loads(add_op_values_str)
        assert add_op_values == [5]


class TestUDFGen_SecureTransferInput_with_SMPC_off(
    TestUDFGenBase, _TestGenerateUDFQueries
):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            transfer=secure_transfer(add_op=True),
            return_type=transfer(),
        )
        def f(transfer):
            return transfer

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="test_secure_transfer_table",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="secure_transfer", dtype=DType.JSON),
                    ]
                ),
                type_=TableType.REMOTE,
            ),
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name()
RETURNS
TABLE("transfer" CLOB)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    import json
    __transfer_strs = _conn.execute("SELECT secure_transfer from test_secure_transfer_table;")["secure_transfer"]
    __transfers = [json.loads(str) for str in __transfer_strs]
    transfer = udfio.secure_transfers_to_merged_dict(__transfers)
    return json.dumps(transfer)
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    *
FROM
    $udf_name();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"transfer" CLOB);'
                ),
            ),
        ]

    @pytest.mark.database
    @pytest.mark.usefixtures(
        "use_globalnode_database",
        "create_secure_transfer_table",
    )
    def test_udf_with_db(
        self,
        concrete_udf_outputs,
        concrete_udf_def,
        concrete_udf_sel,
        globalnode_db_cursor,
    ):
        globalnode_db_cursor.execute(concrete_udf_outputs)
        globalnode_db_cursor.execute(concrete_udf_def)
        globalnode_db_cursor.execute(concrete_udf_sel)
        transfer, *_ = globalnode_db_cursor.execute(
            "SELECT transfer FROM main_output_table_name"
        ).fetchone()
        result = json.loads(transfer)
        assert result == {"sum": 111}


class TestUDFGen_SecureTransferInput_with_SMPC_on(
    TestUDFGenBase, _TestGenerateUDFQueries
):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            transfer=secure_transfer(add_op=True),
            return_type=transfer(),
        )
        def f(transfer):
            return transfer

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            SMPCTablesInfo(
                template=TableInfo(
                    name="test_smpc_template_table",
                    schema_=TableSchema(
                        columns=[
                            ColumnInfo(name="secure_transfer", dtype=DType.JSON),
                        ]
                    ),
                    type_=TableType.NORMAL,
                ),
                add_op_values=TableInfo(
                    name="test_smpc_add_op_values_table",
                    schema_=TableSchema(
                        columns=[
                            ColumnInfo(name="secure_transfer", dtype=DType.JSON),
                        ]
                    ),
                    type_=TableType.NORMAL,
                ),
            )
        ]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name()
RETURNS
TABLE("transfer" CLOB)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    import json
    __template_str = _conn.execute("SELECT secure_transfer from test_smpc_template_table;")["secure_transfer"][0]
    __template = json.loads(__template_str)
    __add_op_values_str = _conn.execute("SELECT secure_transfer from test_smpc_add_op_values_table;")["secure_transfer"][0]
    __add_op_values = json.loads(__add_op_values_str)
    __min_op_values = None
    __max_op_values = None
    __union_op_values = None
    transfer = udfio.construct_secure_transfer_dict(__template,__add_op_values,__min_op_values,__max_op_values,__union_op_values)
    return json.dumps(transfer)
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    *
FROM
    $udf_name();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"transfer" CLOB);'
                ),
            ),
        ]

    @pytest.fixture(scope="class")
    def use_smpc(self):
        return True

    @pytest.mark.database
    @pytest.mark.usefixtures(
        "use_globalnode_database",
        "create_smpc_template_table",
        "create_smpc_add_op_values_table",
    )
    def test_udf_with_db(
        self,
        concrete_udf_outputs,
        concrete_udf_def,
        concrete_udf_sel,
        globalnode_db_cursor,
    ):
        globalnode_db_cursor.execute(concrete_udf_outputs)
        globalnode_db_cursor.execute(concrete_udf_def)
        globalnode_db_cursor.execute(concrete_udf_sel)
        transfer, *_ = globalnode_db_cursor.execute(
            "SELECT transfer FROM main_output_table_name"
        ).fetchone()
        result = json.loads(transfer)
        assert result == {"sum": [100, 200, 300]}


class TestUDFGen_LoggerArgument(TestUDFGenBase, _TestGenerateUDFQueries):
    @pytest.fixture(scope="class")
    def udfregistry(self):
        @udf(
            t=literal(),
            logger=udf_logger(),
            return_type=transfer(),
        )
        def f(t, logger):
            logger.info("Log inside monetdb udf.")
            result = {"num": t}
            return result

        return udf.registry

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [5]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
$udf_name()
RETURNS
TABLE("transfer" CLOB)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    import json
    t = 5
    logger = udfio.get_logger('f_gb47', 'test_udfgenerator')
    logger.info('Log inside monetdb udf.')
    result = {'num': t}
    return json.dumps(result)
}"""

    @pytest.fixture(scope="class")
    def expected_udfsel(self):
        return """\
INSERT INTO $main_output_table_name
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    *
FROM
    $udf_name();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            TableUDFGenResult(
                tablename_placeholder="main_output_table_name",
                drop_query=Template("DROP TABLE IF EXISTS $main_output_table_name;"),
                create_query=Template(
                    'CREATE TABLE $main_output_table_name("node_id" VARCHAR(500),"transfer" CLOB);'
                ),
            )
        ]

    @pytest.fixture(scope="class")
    def request_id(self):
        return "test_udfgenerator"

    @pytest.mark.database
    @pytest.mark.usefixtures("use_globalnode_database")
    def test_udf_with_db(
        self,
        concrete_udf_outputs,
        concrete_udf_def,
        concrete_udf_sel,
        globalnode_db_cursor,
    ):
        globalnode_db_cursor.execute(concrete_udf_outputs)
        globalnode_db_cursor.execute(concrete_udf_def)
        globalnode_db_cursor.execute(concrete_udf_sel)
        _, transfer = globalnode_db_cursor.execute(
            "SELECT * FROM main_output_table_name"
        ).fetchone()
        result = json.loads(transfer)
        assert result == {"num": 5}


# ~~~~~~~~~~~~~~~~~~~~~~ Test SQL Generator ~~~~~~~~~~~~~~~~~~ #


def test_tensor_elementwise_binary_op_1dim():
    tensor_0 = TensorArg(table_name="tens0", dtype=None, ndims=1)
    tensor_1 = TensorArg(table_name="tens1", dtype=None, ndims=1)
    op = TensorBinaryOp.ADD
    expected = """\
SELECT
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
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
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
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
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
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
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
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
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
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
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
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
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
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
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
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
    CAST('$node_id' AS VARCHAR(500)) AS node_id,
    tensor_0.dim0 AS dim0,
    1 - tensor_0.val AS val
FROM
    tens1 AS tensor_0"""
    result = get_tensor_binary_op_template(operand_0, operand_1, op)
    assert result == expected
