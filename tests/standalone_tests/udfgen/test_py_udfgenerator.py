# type: ignore
import json
import pickle
from typing import TypeVar

import pytest

from exareme2.algorithms.exareme2.udfgen import DEFERRED
from exareme2.algorithms.exareme2.udfgen import MIN_ROW_COUNT
from exareme2.algorithms.exareme2.udfgen import literal
from exareme2.algorithms.exareme2.udfgen import merge_transfer
from exareme2.algorithms.exareme2.udfgen import relation
from exareme2.algorithms.exareme2.udfgen import secure_transfer
from exareme2.algorithms.exareme2.udfgen import state
from exareme2.algorithms.exareme2.udfgen import tensor
from exareme2.algorithms.exareme2.udfgen import transfer
from exareme2.algorithms.exareme2.udfgen import udf_logger
from exareme2.algorithms.exareme2.udfgen.decorator import UdfRegistry
from exareme2.algorithms.exareme2.udfgen.decorator import udf
from exareme2.algorithms.exareme2.udfgen.iotypes import LiteralArg
from exareme2.algorithms.exareme2.udfgen.iotypes import MergeTensorType
from exareme2.algorithms.exareme2.udfgen.iotypes import RelationArg
from exareme2.algorithms.exareme2.udfgen.iotypes import StateArg
from exareme2.algorithms.exareme2.udfgen.iotypes import TensorArg
from exareme2.algorithms.exareme2.udfgen.iotypes import TransferArg
from exareme2.algorithms.exareme2.udfgen.py_udfgenerator import (
    FlowArgsToUdfArgsConverter,
)
from exareme2.algorithms.exareme2.udfgen.py_udfgenerator import PyUdfGenerator
from exareme2.algorithms.exareme2.udfgen.py_udfgenerator import UDFBadCall
from exareme2.algorithms.exareme2.udfgen.py_udfgenerator import copy_types_from_udfargs
from exareme2.algorithms.exareme2.udfgen.udfgen_DTOs import UDFGenSMPCResult
from exareme2.algorithms.exareme2.udfgen.udfgen_DTOs import UDFGenTableResult
from exareme2.datatypes import DType
from exareme2.worker.exareme2.udfs.udfs_service import _get_udf_table_creation_queries
from exareme2.worker_communication import ColumnInfo
from exareme2.worker_communication import SMPCTablesInfo
from exareme2.worker_communication import TableInfo
from exareme2.worker_communication import TableSchema
from exareme2.worker_communication import TableType


def test_copy_types_from_udfargs():
    udfgen_args = {
        "a": RelationArg(table_name="A", schema=[("a", int)]),
        "b": TensorArg(table_name="B", dtype=int, ndims=2),
    }
    udfparams = copy_types_from_udfargs(udfgen_args)
    assert udfparams == {
        "a": relation(schema=[("a", int)]),
        "b": tensor(dtype=int, ndims=2),
    }


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
    converter = FlowArgsToUdfArgsConverter()
    result, _ = converter.convert(udfgen_posargs, {})
    assert result == expected_udf_posargs


def test_convert_udfgenargs_to_udfargs_tensor():
    udfgen_posargs = [
        TableInfo(
            name="tab",
            schema_=TableSchema(
                columns=[
                    ColumnInfo(name="dim0", dtype=DType.INT),
                    ColumnInfo(name="dim1", dtype=DType.INT),
                    ColumnInfo(name="val", dtype=DType.FLOAT),
                ]
            ),
            type_=TableType.NORMAL,
        )
    ]
    expected_udf_posargs = [TensorArg(table_name="tab", dtype=float, ndims=2)]
    converter = FlowArgsToUdfArgsConverter()
    result, _ = converter.convert(udfgen_posargs, {})
    assert result == expected_udf_posargs


def test_convert_udfgenargs_to_udfargs_literal():
    udfgen_posargs = [42]
    expected_udf_posargs = [LiteralArg(value=42)]
    converter = FlowArgsToUdfArgsConverter()
    result, _ = converter.convert(udfgen_posargs, {})
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
    converter = FlowArgsToUdfArgsConverter()
    result, _ = converter.convert(udfgen_posargs, {})
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
    converter = FlowArgsToUdfArgsConverter()
    result, _ = converter.convert(udfgen_posargs, {})
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
    converter = FlowArgsToUdfArgsConverter()
    result, _ = converter.convert(udfgen_posargs, {})
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

    converter = FlowArgsToUdfArgsConverter()
    with pytest.raises(UDFBadCall):
        result, _ = converter.convert(udfgen_posargs, {})


class TestUDFGenBase:
    @pytest.fixture(scope="class")
    def funcname(self, udfregistry):
        assert len(udfregistry) == 1, "Multiple entries in udf.registry, expected one."
        return next(iter(udfregistry.keys()))

    @pytest.fixture(scope="class")
    def udfregistry(self):
        udf.registry = UdfRegistry()
        self.define_pyfunc()
        return udf.registry

    @pytest.fixture(scope="function")
    def create_transfer_table(self, globalworker_db_cursor):
        globalworker_db_cursor.execute(
            "CREATE TABLE test_transfer_table(transfer CLOB)"
        )
        globalworker_db_cursor.execute(
            "INSERT INTO test_transfer_table(transfer) VALUES('{\"num\":5}')"
        )

    @pytest.fixture(scope="function")
    def create_state_table(self, globalworker_db_cursor):
        state = pickle.dumps({"num": 5}).hex()
        globalworker_db_cursor.execute("CREATE TABLE test_state_table(state BLOB)")
        insert_state = f"INSERT INTO test_state_table(state) VALUES('{state}')"
        globalworker_db_cursor.execute(insert_state)

    @pytest.fixture(scope="function")
    def create_merge_transfer_table(self, globalworker_db_cursor):
        globalworker_db_cursor.execute(
            "CREATE TABLE test_merge_transfer_table(transfer CLOB)"
        )
        globalworker_db_cursor.execute(
            """INSERT INTO test_merge_transfer_table
                 (transfer)
               VALUES
                 ('{\"num\":5}'),
                 ('{\"num\":10}')"""
        )

    @pytest.fixture(scope="function")
    def create_secure_transfer_table(self, globalworker_db_cursor):
        globalworker_db_cursor.execute(
            "CREATE TABLE test_secure_transfer_table(secure_transfer CLOB)"
        )
        globalworker_db_cursor.execute(
            """INSERT INTO test_secure_transfer_table
                 (secure_transfer)
               VALUES
                 (\'{"sum": {"data": 1, "operation": "sum", "type": "int"}}\'),
                 (\'{"sum": {"data": 10, "operation": "sum", "type": "int"}}\'),
                 (\'{"sum": {"data": 100, "operation": "sum", "type": "int"}}\')"""
        )

    @pytest.fixture(scope="function")
    def create_smpc_template_table_with_sum(self, globalworker_db_cursor):
        globalworker_db_cursor.execute(
            "CREATE TABLE test_smpc_template_table(secure_transfer CLOB)"
        )
        globalworker_db_cursor.execute(
            'INSERT INTO test_smpc_template_table(secure_transfer) VALUES(\'{"sum": {"data": [0,1,2], "operation": "sum", "type": "int"}}\')'
        )

    @pytest.fixture(scope="function")
    def create_smpc_sum_op_values_table(self, globalworker_db_cursor):
        globalworker_db_cursor.execute(
            "CREATE TABLE test_smpc_sum_op_values_table(secure_transfer CLOB)"
        )
        globalworker_db_cursor.execute(
            "INSERT INTO test_smpc_sum_op_values_table(secure_transfer) VALUES('[100,200,300]')"
        )

    @pytest.fixture(scope="function")
    def create_smpc_template_table_with_sum_and_max(self, globalworker_db_cursor):
        globalworker_db_cursor.execute(
            "CREATE TABLE test_smpc_template_table(secure_transfer CLOB)"
        )
        globalworker_db_cursor.execute(
            'INSERT INTO test_smpc_template_table(secure_transfer) VALUES(\'{"sum": {"data": [0,1,2], "operation": "sum", "type": "int"}, '
            '"max": {"data": 0, "operation": "max", "type": "int"}}\')'
        )

    @pytest.fixture(scope="function")
    def create_smpc_max_op_values_table(self, globalworker_db_cursor):
        globalworker_db_cursor.execute(
            "CREATE TABLE test_smpc_max_op_values_table(secure_transfer CLOB)"
        )
        globalworker_db_cursor.execute(
            "INSERT INTO test_smpc_max_op_values_table(secure_transfer) VALUES('[58]')"
        )

    # TODO Should become more dynamic in the future.
    # It should receive a TableInfo object as input and maybe data as well.
    @pytest.fixture(scope="function")
    def create_tensor_table(self, globalworker_db_cursor):
        globalworker_db_cursor.execute(
            "CREATE TABLE tensor_in_db(dim0 INT, dim1 INT, val INT)"
        )
        globalworker_db_cursor.execute(
            """INSERT INTO tensor_in_db
                 (dim0, dim1, val)
               VALUES
                 (0, 0, 3),
                 (0, 1, 4),
                 (0, 2, 7)"""
        )

    @pytest.fixture(scope="function")
    def execute_udf_queries_in_db(
        self,
        globalworker_db_cursor,
        expected_udf_outputs,
        expected_udfdef,
        expected_udfexec,
    ):
        for query in _get_udf_table_creation_queries(expected_udf_outputs):
            globalworker_db_cursor.execute(query)
        globalworker_db_cursor.execute(expected_udfdef)
        globalworker_db_cursor.execute(expected_udfexec)


class TestUDFGen_InvalidUDFArgs_NamesMismatch(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            x=tensor(dtype=int, ndims=1),
            y=tensor(dtype=int, ndims=1),
            z=literal(),
            return_type=relation([("result", int)]),
        )
        def f(x, y, z):
            return x

    def test_get_udf_templates(self, udfregistry, funcname):
        posargs = [TensorArg("table_name", dtype=int, ndims=1)]
        keywordargs = {"z": LiteralArg(1)}
        with pytest.raises(UDFBadCall) as exc:
            PyUdfGenerator(
                udfregistry=udfregistry,
                func_name=funcname,
                flowargs=posargs,
                flowkwargs=keywordargs,
            )
        assert "UDF argument names do not match UDF parameter names" in str(exc)


class TestUDFGen_LoggerArgument_provided_in_pos_args(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            x=tensor(dtype=int, ndims=1),
            logger=udf_logger(),
            return_type=relation([("result", int)]),
        )
        def f(x, logger):
            return x

    def test_get_udf_templates(self, udfregistry, funcname):
        posargs = [TensorArg("table_name", dtype=int, ndims=1), LiteralArg(1)]
        with pytest.raises(UDFBadCall) as exc:
            PyUdfGenerator(
                udfregistry=udfregistry,
                func_name=funcname,
                flowargs=posargs,
                flowkwargs={},
            )
        assert "No argument should be provided for 'UDFLoggerType' parameter" in str(
            exc
        )


class TestUDFGen_LoggerArgument_provided_in_kw_args(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            x=tensor(dtype=int, ndims=1),
            logger=udf_logger(),
            return_type=relation([("result", int)]),
        )
        def f(x, logger):
            return x

    def test_get_udf_templates(self, udfregistry, funcname):
        posargs = [TensorArg("table_name", dtype=int, ndims=1)]
        keywordargs = {"logger": LiteralArg(1)}
        with pytest.raises(UDFBadCall) as exc:
            PyUdfGenerator(
                udfregistry=udfregistry,
                func_name=funcname,
                flowargs=posargs,
                flowkwargs=keywordargs,
            )
        assert "No argument should be provided for 'UDFLoggerType' parameter" in str(
            exc
        )


class TestUDFGen_InvalidUDFArgs_TransferTableInStateArgument(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            transfers=state(),
            state=state(),
            return_type=transfer(),
        )
        def f(transfers, state):
            result = {"num": sum}
            return result

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
            PyUdfGenerator(
                udfregistry=udfregistry,
                func_name=funcname,
                flowargs=posargs,
                flowkwargs={},
            )
        assert "should be of type" in str(exc)


class TestUDFGen_InvalidUDFArgs_TensorTableInTransferArgument(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            transfers=transfer(),
            state=state(),
            return_type=transfer(),
        )
        def f(transfers, state):
            result = {"num": sum}
            return result

    def test_get_udf_templates(self, udfregistry, funcname):
        posargs = [
            TableInfo(
                name="tensor_in_db",
                schema_=TableSchema(
                    columns=[
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
            PyUdfGenerator(
                udfregistry=udfregistry,
                func_name=funcname,
                flowargs=posargs,
                flowkwargs={},
                smpc_used=True,
            )
        assert "should be of type" in str(exc)


class TestUDFGen_Invalid_SMPCUDFInput_To_Transfer_Type(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            transfer=transfer(),
            return_type=transfer(),
        )
        def f(transfer):
            result = {"num": sum}
            return result

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
                sum_op=TableInfo(
                    name="test_smpc_sum_op_values_table",
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
            PyUdfGenerator(
                udfregistry=udfregistry,
                func_name=funcname,
                flowargs=posargs,
                flowkwargs={},
                smpc_used=True,
            )
        assert "should be of type" in str(exc)


class TestUDFGen_Invalid_TableInfoArgs_To_SecureTransferType(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            transfer=secure_transfer(sum_op=True),
            return_type=transfer(),
        )
        def f(transfer):
            result = {"num": sum}
            return result

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

        with pytest.raises(UDFBadCall) as exc:
            PyUdfGenerator(
                udfregistry=udfregistry,
                func_name=funcname,
                flowargs=posargs,
                flowkwargs={},
                smpc_used=True,
            )
        assert "When smpc is used SecureTransferArg should not be" in str(exc)


class TestUDFGen_Invalid_SMPCUDFInput_with_SMPC_off(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            transfer=secure_transfer(sum_op=True),
            return_type=transfer(),
        )
        def f(transfer):
            result = {"num": sum}
            return result

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
                sum_op_values=TableInfo(
                    name="test_smpc_sum_op_values_table",
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
            PyUdfGenerator(
                udfregistry=udfregistry,
                func_name=funcname,
                flowargs=posargs,
                flowkwargs={},
            )
        assert "SMPC is not used, " in str(exc)


class TestUDFGen_InvalidUDFArgs_InconsistentTypeVars(TestUDFGenBase):
    def define_pyfunc(self):
        T = TypeVar("T")

        @udf(
            x=tensor(dtype=T, ndims=1),
            y=tensor(dtype=T, ndims=1),
            return_type=tensor(dtype=T, ndims=1),
        )
        def f(x, y):
            return x

    def test_get_udf_templates(self, udfregistry, funcname):
        posargs = [
            TableInfo(
                name="table_name1",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="dim0", dtype=DType.INT),
                        ColumnInfo(name="val", dtype=DType.INT),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
            TableInfo(
                name="table_name1",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="dim0", dtype=DType.FLOAT),
                        ColumnInfo(name="val", dtype=DType.FLOAT),
                    ]
                ),
                type_=TableType.NORMAL,
            ),
        ]

        keywordargs = {}
        with pytest.raises(ValueError) as e:
            PyUdfGenerator(
                udfregistry=udfregistry,
                func_name=funcname,
                flowargs=posargs,
                flowkwargs=keywordargs,
            ).get_definition(udf_name="")
        err_msg, *_ = e.value.args
        assert "inconsistent mappings" in err_msg


class TestUDFGen_TensorToTensor(TestUDFGenBase):
    def define_pyfunc(self):
        T = TypeVar("T")

        @udf(x=tensor(dtype=T, ndims=2), return_type=tensor(dtype=DType.FLOAT, ndims=2))
        def f(x):
            result = x
            return result

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="tensor_in_db",
                schema_=TableSchema(
                    columns=[
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
__udf("x_dim0" INT,"x_dim1" INT,"x_val" INT)
RETURNS
TABLE("dim0" INT,"dim1" INT,"val" DOUBLE)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    x = udfio.from_tensor_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['dim0', 'dim1', 'val'], ['x_dim0', 'x_dim1', 'x_val'])})
    result = x
    return udfio.as_tensor_table(numpy.array(result))
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf((
        SELECT
            tensor_in_db."dim0",
            tensor_in_db."dim1",
            tensor_in_db."val"
        FROM
            tensor_in_db
    ));"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("dim0", DType.INT),
                    ("dim1", DType.INT),
                    ("val", DType.FLOAT),
                ],
                create_query='CREATE TABLE __main("dim0" INT,"dim1" INT,"val" DOUBLE);',
            )
        ]

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
            smpc_used=False,
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_TensorParameterWithCapitalLetter(TestUDFGenBase):
    def define_pyfunc(self):
        T = TypeVar("T")

        @udf(X=tensor(dtype=T, ndims=2), return_type=tensor(dtype=DType.FLOAT, ndims=2))
        def f(X):
            result = X
            return result

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="tensor_in_db",
                schema_=TableSchema(
                    columns=[
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
__udf("X_dim0" INT,"X_dim1" INT,"X_val" INT)
RETURNS
TABLE("dim0" INT,"dim1" INT,"val" DOUBLE)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    X = udfio.from_tensor_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['dim0', 'dim1', 'val'], ['X_dim0', 'X_dim1', 'X_val'])})
    result = X
    return udfio.as_tensor_table(numpy.array(result))
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf((
        SELECT
            tensor_in_db."dim0",
            tensor_in_db."dim1",
            tensor_in_db."val"
        FROM
            tensor_in_db
    ));"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("dim0", DType.INT),
                    ("dim1", DType.INT),
                    ("val", DType.FLOAT),
                ],
                create_query='CREATE TABLE __main("dim0" INT,"dim1" INT,"val" DOUBLE);',
            )
        ]

    @pytest.mark.slow
    @pytest.mark.database
    @pytest.mark.usefixtures("use_globalworker_database", "create_tensor_table")
    def test_udf_with_db(
        self,
        execute_udf_queries_in_db,
        globalworker_db_cursor,
        create_tensor_table,
    ):
        db = globalworker_db_cursor

        output_table_values = db.execute("SELECT * FROM __main").fetchall()

        assert output_table_values == [
            (0, 0, 3.0),
            (0, 1, 4.0),
            (0, 2, 7.0),
        ]

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
            smpc_used=False,
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_RelationToTensor(TestUDFGenBase):
    def define_pyfunc(self):
        S = TypeVar("S")

        @udf(r=relation(schema=S), return_type=tensor(dtype=DType.FLOAT, ndims=2))
        def f(r):
            result = r
            return result

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="rel_in_db",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.INT),
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
__udf("r_row_id" INT,"r_col0" INT,"r_col1" DOUBLE,"r_col2" VARCHAR(500))
RETURNS
TABLE("dim0" INT,"dim1" INT,"val" DOUBLE)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    r = udfio.from_relational_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['row_id', 'col0', 'col1', 'col2'], ['r_row_id', 'r_col0', 'r_col1', 'r_col2'])}, 'row_id')
    result = r
    return udfio.as_tensor_table(numpy.array(result))
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf((
        SELECT
            rel_in_db."row_id",
            rel_in_db."col0",
            rel_in_db."col1",
            rel_in_db."col2"
        FROM
            rel_in_db
    ));"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("dim0", DType.INT),
                    ("dim1", DType.INT),
                    ("val", DType.FLOAT),
                ],
                create_query='CREATE TABLE __main("dim0" INT,"dim1" INT,"val" DOUBLE);',
            )
        ]

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
            smpc_used=False,
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_2RelationsToTensor(TestUDFGenBase):
    def define_pyfunc(self):
        S = TypeVar("S")

        @udf(
            r1=relation(schema=S),
            r2=relation(schema=S),
            return_type=tensor(dtype=DType.FLOAT, ndims=2),
        )
        def f(r1, r2):
            result = r1
            return result

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="rel1_in_db",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.INT),
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
                        ColumnInfo(name="row_id", dtype=DType.INT),
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
__udf("r1_row_id" INT,"r1_col0" INT,"r1_col1" DOUBLE,"r1_col2" VARCHAR(500),"r2_row_id" INT,"r2_col4" INT,"r2_col5" DOUBLE,"r2_col6" VARCHAR(500))
RETURNS
TABLE("dim0" INT,"dim1" INT,"val" DOUBLE)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    r1 = udfio.from_relational_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['row_id', 'col0', 'col1', 'col2'], ['r1_row_id', 'r1_col0', 'r1_col1', 'r1_col2'])}, 'row_id')
    r2 = udfio.from_relational_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['row_id', 'col4', 'col5', 'col6'], ['r2_row_id', 'r2_col4', 'r2_col5', 'r2_col6'])}, 'row_id')
    result = r1
    return udfio.as_tensor_table(numpy.array(result))
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf((
        SELECT
            rel1_in_db."row_id",
            rel1_in_db."col0",
            rel1_in_db."col1",
            rel1_in_db."col2",
            rel2_in_db."row_id",
            rel2_in_db."col4",
            rel2_in_db."col5",
            rel2_in_db."col6"
        FROM
            rel1_in_db,
            rel2_in_db
        WHERE
            rel1_in_db."row_id"=rel2_in_db."row_id"
    ));"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("dim0", DType.INT),
                    ("dim1", DType.INT),
                    ("val", DType.FLOAT),
                ],
                create_query='CREATE TABLE __main("dim0" INT,"dim1" INT,"val" DOUBLE);',
            )
        ]

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
            smpc_used=False,
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_3RelationsToTensor(TestUDFGenBase):
    def define_pyfunc(self):
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

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="rel1_in_db",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.INT),
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
                        ColumnInfo(name="row_id", dtype=DType.INT),
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
                        ColumnInfo(name="row_id", dtype=DType.INT),
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
__udf("r1_row_id" INT,"r1_col0" INT,"r1_col1" DOUBLE,"r1_col2" VARCHAR(500),"r2_row_id" INT,"r2_col4" INT,"r2_col5" DOUBLE,"r2_col6" VARCHAR(500),"r3_row_id" INT,"r3_col8" INT,"r3_col9" DOUBLE,"r3_col10" VARCHAR(500))
RETURNS
TABLE("dim0" INT,"dim1" INT,"val" DOUBLE)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    r1 = udfio.from_relational_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['row_id', 'col0', 'col1', 'col2'], ['r1_row_id', 'r1_col0', 'r1_col1', 'r1_col2'])}, 'row_id')
    r2 = udfio.from_relational_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['row_id', 'col4', 'col5', 'col6'], ['r2_row_id', 'r2_col4', 'r2_col5', 'r2_col6'])}, 'row_id')
    r3 = udfio.from_relational_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['row_id', 'col8', 'col9', 'col10'], ['r3_row_id', 'r3_col8', 'r3_col9', 'r3_col10'])}, 'row_id')
    result = r1
    return udfio.as_tensor_table(numpy.array(result))
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf((
        SELECT
            rel1_in_db."row_id",
            rel1_in_db."col0",
            rel1_in_db."col1",
            rel1_in_db."col2",
            rel2_in_db."row_id",
            rel2_in_db."col4",
            rel2_in_db."col5",
            rel2_in_db."col6",
            rel3_in_db."row_id",
            rel3_in_db."col8",
            rel3_in_db."col9",
            rel3_in_db."col10"
        FROM
            rel1_in_db,
            rel2_in_db,
            rel3_in_db
        WHERE
            rel1_in_db."row_id"=rel2_in_db."row_id" AND
            rel1_in_db."row_id"=rel3_in_db."row_id"
    ));"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("dim0", DType.INT),
                    ("dim1", DType.INT),
                    ("val", DType.FLOAT),
                ],
                create_query='CREATE TABLE __main("dim0" INT,"dim1" INT,"val" DOUBLE);',
            )
        ]

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
            smpc_used=False,
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_2SameRelationsToTensor(TestUDFGenBase):
    def define_pyfunc(self):
        S = TypeVar("S")

        @udf(
            r1=relation(schema=S),
            r2=relation(schema=S),
            return_type=tensor(dtype=DType.FLOAT, ndims=2),
        )
        def f(r1, r2):
            result = r1
            return result

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [
            TableInfo(
                name="rel1_in_db",
                schema_=TableSchema(
                    columns=[
                        ColumnInfo(name="row_id", dtype=DType.INT),
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
                        ColumnInfo(name="row_id", dtype=DType.INT),
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
__udf("r1_row_id" INT,"r1_col0" INT,"r1_col1" DOUBLE,"r1_col2" VARCHAR(500),"r2_row_id" INT,"r2_col0" INT,"r2_col1" DOUBLE,"r2_col2" VARCHAR(500))
RETURNS
TABLE("dim0" INT,"dim1" INT,"val" DOUBLE)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    r1 = udfio.from_relational_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['row_id', 'col0', 'col1', 'col2'], ['r1_row_id', 'r1_col0', 'r1_col1', 'r1_col2'])}, 'row_id')
    r2 = udfio.from_relational_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['row_id', 'col0', 'col1', 'col2'], ['r2_row_id', 'r2_col0', 'r2_col1', 'r2_col2'])}, 'row_id')
    result = r1
    return udfio.as_tensor_table(numpy.array(result))
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf((
        SELECT
            rel1_in_db."row_id",
            rel1_in_db."col0",
            rel1_in_db."col1",
            rel1_in_db."col2",
            rel1_in_db."row_id",
            rel1_in_db."col0",
            rel1_in_db."col1",
            rel1_in_db."col2"
        FROM
            rel1_in_db
        WHERE
            rel1_in_db."row_id"=rel1_in_db."row_id"
    ));"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("dim0", DType.INT),
                    ("dim1", DType.INT),
                    ("val", DType.FLOAT),
                ],
                create_query='CREATE TABLE __main("dim0" INT,"dim1" INT,"val" DOUBLE);',
            )
        ]

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_TensorToRelation(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            x=tensor(dtype=int, ndims=1),
            return_type=relation(schema=[("ci", int), ("cf", float)]),
        )
        def f(x):
            return x

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
__udf("x_dim0" INT,"x_val" INT)
RETURNS
TABLE("ci" INT,"cf" DOUBLE)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    x = udfio.from_tensor_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['dim0', 'val'], ['x_dim0', 'x_val'])})
    return udfio.as_relational_table(x, 'row_id')
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf((
        SELECT
            tensor_in_db."dim0",
            tensor_in_db."val"
        FROM
            tensor_in_db
    ));"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("ci", DType.INT),
                    ("cf", DType.FLOAT),
                ],
                create_query='CREATE TABLE __main("ci" INT,"cf" DOUBLE);',
            )
        ]

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_LiteralArgument(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            x=tensor(dtype=DType.INT, ndims=1),
            v=literal(),
            return_type=relation([("result", int)]),
        )
        def f(x, v):
            result = v
            return result

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
__udf("x_dim0" INT,"x_val" INT)
RETURNS
TABLE("result" INT)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    x = udfio.from_tensor_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['dim0', 'val'], ['x_dim0', 'x_val'])})
    v = 42
    result = v
    return udfio.as_relational_table(result, 'row_id')
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf((
        SELECT
            the_table."dim0",
            the_table."val"
        FROM
            the_table
    ));"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("result", DType.INT),
                ],
                create_query='CREATE TABLE __main("result" INT);',
            )
        ]

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_ManyLiteralArguments(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            x=tensor(dtype=DType.INT, ndims=1),
            v=literal(),
            w=literal(),
            return_type=relation([("result", int)]),
        )
        def f(x, v, w):
            result = v + w
            return result

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
__udf("x_dim0" INT,"x_val" INT)
RETURNS
TABLE("result" INT)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    x = udfio.from_tensor_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['dim0', 'val'], ['x_dim0', 'x_val'])})
    v = 42
    w = 24
    result = v + w
    return udfio.as_relational_table(result, 'row_id')
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf((
        SELECT
            tensor_in_db."dim0",
            tensor_in_db."val"
        FROM
            tensor_in_db
    ));"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("result", DType.INT),
                ],
                create_query='CREATE TABLE __main("result" INT);',
            )
        ]

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_NoArguments(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            return_type=tensor(dtype=int, ndims=1),
        )
        def f():
            x = [1, 2, 3]
            return x

    @pytest.fixture(scope="class")
    def positional_args(self):
        return []

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
__udf()
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
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("dim0", DType.INT),
                    ("val", DType.INT),
                ],
                create_query='CREATE TABLE __main("dim0" INT,"val" INT);',
            )
        ]

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_RelationIncludeRowId(TestUDFGenBase):
    def define_pyfunc(self):
        S = TypeVar("S")

        @udf(r=relation(schema=S), return_type=tensor(dtype=DType.FLOAT, ndims=2))
        def f(r):
            result = r
            return result

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
__udf("r_row_id" INT,"r_c0" INT,"r_c1" DOUBLE,"r_c2" VARCHAR(500))
RETURNS
TABLE("dim0" INT,"dim1" INT,"val" DOUBLE)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    r = udfio.from_relational_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['row_id', 'c0', 'c1', 'c2'], ['r_row_id', 'r_c0', 'r_c1', 'r_c2'])}, 'row_id')
    result = r
    return udfio.as_tensor_table(numpy.array(result))
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf((
        SELECT
            rel_in_db."row_id",
            rel_in_db."c0",
            rel_in_db."c1",
            rel_in_db."c2"
        FROM
            rel_in_db
    ));"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("dim0", DType.INT),
                    ("dim1", DType.INT),
                    ("val", DType.FLOAT),
                ],
                create_query='CREATE TABLE __main("dim0" INT,"dim1" INT,"val" DOUBLE);',
            )
        ]

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_RelationExcludeNodeid(TestUDFGenBase):
    def define_pyfunc(self):
        S = TypeVar("S")

        @udf(r=relation(schema=S), return_type=tensor(dtype=DType.FLOAT, ndims=2))
        def f(r):
            result = r
            return result

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
__udf("r_row_id" INT,"r_c0" INT,"r_c1" DOUBLE,"r_c2" VARCHAR(500))
RETURNS
TABLE("dim0" INT,"dim1" INT,"val" DOUBLE)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    r = udfio.from_relational_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['row_id', 'c0', 'c1', 'c2'], ['r_row_id', 'r_c0', 'r_c1', 'r_c2'])}, 'row_id')
    result = r
    return udfio.as_tensor_table(numpy.array(result))
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf((
        SELECT
            rel_in_db."row_id",
            rel_in_db."c0",
            rel_in_db."c1",
            rel_in_db."c2"
        FROM
            rel_in_db
    ));"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("dim0", DType.INT),
                    ("dim1", DType.INT),
                    ("val", DType.FLOAT),
                ],
                create_query='CREATE TABLE __main("dim0" INT,"dim1" INT,"val" DOUBLE);',
            )
        ]

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_UnknownReturnDimensions(TestUDFGenBase):
    def define_pyfunc(self):
        T = TypeVar("S")
        N = TypeVar("N")

        @udf(t=tensor(dtype=T, ndims=N), return_type=tensor(dtype=T, ndims=N))
        def f(t):
            result = t
            return result

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
__udf("t_dim0" INT,"t_dim1" INT,"t_val" INT)
RETURNS
TABLE("dim0" INT,"dim1" INT,"val" INT)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    t = udfio.from_tensor_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['dim0', 'dim1', 'val'], ['t_dim0', 't_dim1', 't_val'])})
    result = t
    return udfio.as_tensor_table(numpy.array(result))
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf((
        SELECT
            tens_in_db."dim0",
            tens_in_db."dim1",
            tens_in_db."val"
        FROM
            tens_in_db
    ));"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("dim0", DType.INT),
                    ("dim1", DType.INT),
                    ("val", DType.INT),
                ],
                create_query='CREATE TABLE __main("dim0" INT,"dim1" INT,"val" INT);',
            )
        ]

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_TwoTensors1DReturnTable(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            x=tensor(dtype=int, ndims=1),
            y=tensor(dtype=int, ndims=1),
            return_type=tensor(dtype=int, ndims=1),
        )
        def f(x, y):
            result = x - y
            return result

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
__udf("x_dim0" INT,"x_val" INT,"y_dim0" INT,"y_val" INT)
RETURNS
TABLE("dim0" INT,"val" INT)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    x = udfio.from_tensor_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['dim0', 'val'], ['x_dim0', 'x_val'])})
    y = udfio.from_tensor_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['dim0', 'val'], ['y_dim0', 'y_val'])})
    result = x - y
    return udfio.as_tensor_table(numpy.array(result))
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf((
        SELECT
            tens0."dim0",
            tens0."val",
            tens1."dim0",
            tens1."val"
        FROM
            tens0,
            tens1
        WHERE
            tens0."dim0"=tens1."dim0"
    ));"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("dim0", DType.INT),
                    ("val", DType.INT),
                ],
                create_query='CREATE TABLE __main("dim0" INT,"val" INT);',
            )
        ]

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_ThreeTensors1DReturnTable(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            x=tensor(dtype=int, ndims=1),
            y=tensor(dtype=int, ndims=1),
            z=tensor(dtype=int, ndims=1),
            return_type=tensor(dtype=int, ndims=1),
        )
        def f(x, y, z):
            result = x + y + z
            return result

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
__udf("x_dim0" INT,"x_val" INT,"y_dim0" INT,"y_val" INT,"z_dim0" INT,"z_val" INT)
RETURNS
TABLE("dim0" INT,"val" INT)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    x = udfio.from_tensor_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['dim0', 'val'], ['x_dim0', 'x_val'])})
    y = udfio.from_tensor_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['dim0', 'val'], ['y_dim0', 'y_val'])})
    z = udfio.from_tensor_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['dim0', 'val'], ['z_dim0', 'z_val'])})
    result = x + y + z
    return udfio.as_tensor_table(numpy.array(result))
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf((
        SELECT
            tens0."dim0",
            tens0."val",
            tens1."dim0",
            tens1."val",
            tens2."dim0",
            tens2."val"
        FROM
            tens0,
            tens1,
            tens2
        WHERE
            tens0."dim0"=tens1."dim0" AND
            tens0."dim0"=tens2."dim0"
    ));"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("dim0", DType.INT),
                    ("val", DType.INT),
                ],
                create_query='CREATE TABLE __main("dim0" INT,"val" INT);',
            )
        ]

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_ThreeTensors2DReturnTable(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            x=tensor(dtype=int, ndims=2),
            y=tensor(dtype=int, ndims=2),
            z=tensor(dtype=int, ndims=2),
            return_type=tensor(dtype=int, ndims=2),
        )
        def f(x, y, z):
            result = x + y + z
            return result

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
__udf("x_dim0" INT,"x_dim1" INT,"x_val" INT,"y_dim0" INT,"y_dim1" INT,"y_val" INT,"z_dim0" INT,"z_dim1" INT,"z_val" INT)
RETURNS
TABLE("dim0" INT,"dim1" INT,"val" INT)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    x = udfio.from_tensor_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['dim0', 'dim1', 'val'], ['x_dim0', 'x_dim1', 'x_val'])})
    y = udfio.from_tensor_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['dim0', 'dim1', 'val'], ['y_dim0', 'y_dim1', 'y_val'])})
    z = udfio.from_tensor_table({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['dim0', 'dim1', 'val'], ['z_dim0', 'z_dim1', 'z_val'])})
    result = x + y + z
    return udfio.as_tensor_table(numpy.array(result))
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf((
        SELECT
            tens0."dim0",
            tens0."dim1",
            tens0."val",
            tens1."dim0",
            tens1."dim1",
            tens1."val",
            tens2."dim0",
            tens2."dim1",
            tens2."val"
        FROM
            tens0,
            tens1,
            tens2
        WHERE
            tens0."dim0"=tens1."dim0" AND
            tens0."dim1"=tens1."dim1" AND
            tens0."dim0"=tens2."dim0" AND
            tens0."dim1"=tens2."dim1"
    ));"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("dim0", DType.INT),
                    ("dim1", DType.INT),
                    ("val", DType.INT),
                ],
                create_query='CREATE TABLE __main("dim0" INT,"dim1" INT,"val" INT);',
            )
        ]

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_MergeTensor(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            xs=MergeTensorType(dtype=int, ndims=1),
            return_type=tensor(dtype=int, ndims=1),
        )
        def sum_tensors(xs):
            x = sum(xs)
            return x

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
__udf("xs_dim0" INT,"xs_val" INT)
RETURNS
TABLE("dim0" INT,"val" INT)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    xs = udfio.merge_tensor_to_list({name: _columns[name_w_prefix] for name, name_w_prefix in zip(['dim0', 'val'], ['xs_dim0', 'xs_val'])})
    x = sum(xs)
    return udfio.as_tensor_table(numpy.array(x))
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf((
        SELECT
            merge_table."dim0",
            merge_table."val"
        FROM
            merge_table
    ));"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("dim0", DType.INT),
                    ("val", DType.INT),
                ],
                create_query='CREATE TABLE __main("dim0" INT,"val" INT);',
            )
        ]

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_StateReturnType(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(t=literal(), return_type=state())
        def f(t):
            result = {"num": 5}
            return result

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [5]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
__udf()
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
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("state", DType.BINARY),
                ],
                create_query='CREATE TABLE __main("state" BLOB);',
            )
        ]

    @pytest.mark.slow
    @pytest.mark.database
    @pytest.mark.usefixtures("use_globalworker_database")
    def test_udf_with_db(
        self,
        execute_udf_queries_in_db,
        globalworker_db_cursor,
    ):
        db = globalworker_db_cursor

        [state] = db.execute("SELECT * FROM __main").fetchone()
        result = pickle.loads(state)

        assert result == {"num": 5}

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_StateInputandReturnType(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            t=literal(),
            prev_state=state(),
            return_type=state(),
        )
        def f(t, prev_state):
            prev_state["num"] = prev_state["num"] + t
            return prev_state

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
__udf()
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
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("state", DType.BINARY),
                ],
                create_query='CREATE TABLE __main("state" BLOB);',
            )
        ]

    @pytest.mark.slow
    @pytest.mark.database
    @pytest.mark.usefixtures("use_globalworker_database", "create_state_table")
    def test_udf_with_db(
        self,
        execute_udf_queries_in_db,
        globalworker_db_cursor,
    ):
        db = globalworker_db_cursor

        [state] = db.execute("SELECT * FROM __main").fetchone()
        result = pickle.loads(state)

        assert result == {"num": 10}

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_TransferReturnType(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(t=literal(), return_type=transfer())
        def f(t):
            result = {"num": t, "list_of_nums": [t, t, t]}
            return result

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [5]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
__udf()
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
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("transfer", DType.JSON),
                ],
                create_query='CREATE TABLE __main("transfer" CLOB);',
                share=True,
            )
        ]

    @pytest.mark.slow
    @pytest.mark.database
    @pytest.mark.usefixtures("use_globalworker_database")
    def test_udf_with_db(
        self,
        execute_udf_queries_in_db,
        globalworker_db_cursor,
    ):
        db = globalworker_db_cursor

        [transfer] = db.execute("SELECT * FROM __main").fetchone()
        result = json.loads(transfer)

        assert result == {"num": 5, "list_of_nums": [5, 5, 5]}

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_TransferInputAndReturnType(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            t=literal(),
            transfer=transfer(),
            return_type=transfer(),
        )
        def f(t, transfer):
            transfer["num"] = transfer["num"] + t
            return transfer

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
__udf()
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
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("transfer", DType.JSON),
                ],
                create_query='CREATE TABLE __main("transfer" CLOB);',
                share=True,
            )
        ]

    @pytest.mark.slow
    @pytest.mark.database
    @pytest.mark.usefixtures("use_globalworker_database", "create_transfer_table")
    def test_udf_with_db(
        self,
        execute_udf_queries_in_db,
        globalworker_db_cursor,
    ):
        db = globalworker_db_cursor

        [transfer] = db.execute("SELECT * FROM __main").fetchone()
        result = json.loads(transfer)

        assert result == {"num": 10}

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_TransferInputAndStateReturnType(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            t=literal(),
            transfer=transfer(),
            return_type=state(),
        )
        def f(t, transfer):
            transfer["num"] = transfer["num"] + t
            return transfer

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
__udf()
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
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("state", DType.BINARY),
                ],
                create_query='CREATE TABLE __main("state" BLOB);',
            )
        ]

    @pytest.mark.slow
    @pytest.mark.database
    @pytest.mark.usefixtures("use_globalworker_database", "create_transfer_table")
    def test_udf_with_db(
        self,
        execute_udf_queries_in_db,
        globalworker_db_cursor,
    ):
        db = globalworker_db_cursor

        [state] = db.execute("SELECT * FROM __main").fetchone()
        result = pickle.loads(state)

        assert result == {"num": 10}

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_TransferAndStateInputandStateReturnType(TestUDFGenBase):
    def define_pyfunc(self):
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
__udf()
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
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("state", DType.BINARY),
                ],
                create_query='CREATE TABLE __main("state" BLOB);',
            )
        ]

    @pytest.mark.slow
    @pytest.mark.database
    @pytest.mark.usefixtures(
        "use_globalworker_database",
        "create_transfer_table",
        "create_state_table",
    )
    def test_udf_with_db(
        self,
        execute_udf_queries_in_db,
        globalworker_db_cursor,
    ):
        db = globalworker_db_cursor

        [state] = db.execute("SELECT * FROM __main").fetchone()
        result = pickle.loads(state)

        assert result == {"num": 15}

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_MergeTransferAndStateInputandTransferReturnType(TestUDFGenBase):
    def define_pyfunc(self):
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
__udf()
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
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("transfer", DType.JSON),
                ],
                create_query='CREATE TABLE __main("transfer" CLOB);',
                share=True,
            )
        ]

    @pytest.mark.slow
    @pytest.mark.database
    @pytest.mark.usefixtures(
        "use_globalworker_database",
        "create_merge_transfer_table",
        "create_state_table",
    )
    def test_udf_with_db(
        self,
        execute_udf_queries_in_db,
        globalworker_db_cursor,
    ):
        db = globalworker_db_cursor

        [transfer] = db.execute("SELECT * FROM __main").fetchone()
        result = json.loads(transfer)

        assert result == {"num": 20}

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_LocalStepLogic(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            state=state(),
            transfer=transfer(),
            return_type=[state(), transfer()],
        )
        def f(state, transfer):
            result1 = {"num": transfer["num"] + state["num"]}
            result2 = {"num": transfer["num"] * state["num"]}
            return result1, result2

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
__udf()
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
    _conn.execute(f"INSERT INTO __lt0 VALUES ('{json.dumps(result2)}');")
    return pickle.dumps(result1)
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("state", DType.BINARY),
                ],
                create_query='CREATE TABLE __main("state" BLOB);',
            ),
            UDFGenTableResult(
                table_name="__lt0",
                table_schema=[
                    ("transfer", DType.JSON),
                ],
                create_query='CREATE TABLE __lt0("transfer" CLOB);',
                share=True,
            ),
        ]

    @pytest.mark.slow
    @pytest.mark.database
    @pytest.mark.usefixtures(
        "use_globalworker_database",
        "create_transfer_table",
        "create_state_table",
    )
    def test_udf_with_db(
        self,
        execute_udf_queries_in_db,
        globalworker_db_cursor,
    ):
        db = globalworker_db_cursor

        [state] = db.execute("SELECT state FROM __main").fetchone()
        result1 = pickle.loads(state)

        assert result1 == {"num": 10}

        [transfer] = db.execute("SELECT transfer FROM __lt0").fetchone()
        result2 = json.loads(transfer)

        assert result2 == {"num": 25}

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(
            udf_name="__udf", output_table_names=["__main", "__lt0"]
        )
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(
            udf_name="__udf", output_table_names=["__main", "__lt0"]
        )
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main", "__lt0"])
        assert len(results) == 2
        assert results[0] == expected_udf_outputs[0]
        assert results[1] == expected_udf_outputs[1]


class TestUDFGen_LocalStepLogic_Transfer_first_input_and_output(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            transfer=transfer(),
            state=state(),
            return_type=[transfer(), state()],
        )
        def f(transfer, state):
            result1 = {"num": transfer["num"] + state["num"]}
            result2 = {"num": transfer["num"] * state["num"]}
            return result1, result2

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
__udf()
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
    _conn.execute(f"INSERT INTO __lt0 VALUES ('{pickle.dumps(result2).hex()}');")
    return json.dumps(result1)
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("transfer", DType.JSON),
                ],
                create_query='CREATE TABLE __main("transfer" CLOB);',
                share=True,
            ),
            UDFGenTableResult(
                table_name="__lt0",
                table_schema=[
                    ("state", DType.BINARY),
                ],
                create_query='CREATE TABLE __lt0("state" BLOB);',
            ),
        ]

    @pytest.mark.slow
    @pytest.mark.database
    @pytest.mark.usefixtures(
        "use_globalworker_database",
        "create_transfer_table",
        "create_state_table",
    )
    def test_udf_with_globalnode_db_cursor(
        self,
        expected_udf_outputs,
        expected_udfdef,
        expected_udfexec,
        globalworker_db_cursor,
    ):
        db = globalworker_db_cursor
        for output in expected_udf_outputs:
            db.execute(output.create_query)
        db.execute(expected_udfdef)
        db.execute(expected_udfexec)

        [transfer_] = db.execute("SELECT transfer FROM __main").fetchone()
        result1 = json.loads(transfer_)

        assert result1 == {"num": 10}

        [state_] = db.execute("SELECT state FROM __lt0").fetchone()
        result2 = pickle.loads(state_)

        assert result2 == {"num": 25}

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(
            udf_name="__udf", output_table_names=["__main", "__lt0"]
        )
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(
            udf_name="__udf", output_table_names=["__main", "__lt0"]
        )
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main", "__lt0"])
        assert len(results) == 2
        assert results[0] == expected_udf_outputs[0]
        assert results[1] == expected_udf_outputs[1]


class TestUDFGen_GlobalStepLogic(TestUDFGenBase):
    def define_pyfunc(self):
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
__udf()
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
    _conn.execute(f"INSERT INTO __lt0 VALUES ('{json.dumps(result2)}');")
    return pickle.dumps(result1)
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("state", DType.BINARY),
                ],
                create_query='CREATE TABLE __main("state" BLOB);',
            ),
            UDFGenTableResult(
                table_name="__lt0",
                table_schema=[
                    ("transfer", DType.JSON),
                ],
                create_query='CREATE TABLE __lt0("transfer" CLOB);',
                share=True,
            ),
        ]

    @pytest.mark.slow
    @pytest.mark.database
    @pytest.mark.usefixtures(
        "use_globalworker_database",
        "create_merge_transfer_table",
        "create_state_table",
    )
    def test_udf_with_db(
        self,
        execute_udf_queries_in_db,
        globalworker_db_cursor,
    ):
        db = globalworker_db_cursor

        [state_] = db.execute("SELECT state FROM __main").fetchone()
        result1 = pickle.loads(state_)

        assert result1 == {"num": 20}

        [transfer_] = db.execute("SELECT transfer FROM __lt0").fetchone()
        result2 = json.loads(transfer_)

        assert result2 == {"num": 75}

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(
            udf_name="__udf", output_table_names=["__main", "__lt0"]
        )
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(
            udf_name="__udf", output_table_names=["__main", "__lt0"]
        )
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main", "__lt0"])
        assert len(results) == 2
        assert results[0] == expected_udf_outputs[0]
        assert results[1] == expected_udf_outputs[1]


class TestUDFGen_SecureTransferOutput_with_SMPC_off(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            state=state(),
            return_type=secure_transfer(sum_op=True, min_op=True, max_op=True),
        )
        def f(state):
            result = {
                "sum": {"data": state["num"], "operation": "sum", "type": "int"},
                "min": {"data": state["num"], "operation": "min", "type": "int"},
                "max": {"data": state["num"], "operation": "max", "type": "int"},
            }
            return result

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
__udf()
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
    result = {'sum': {'data': state['num'], 'operation': 'sum', 'type': 'int'},
        'min': {'data': state['num'], 'operation': 'min', 'type': 'int'}, 'max':
        {'data': state['num'], 'operation': 'max', 'type': 'int'}}
    return json.dumps(result)
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("secure_transfer", DType.JSON),
                ],
                create_query='CREATE TABLE __main("secure_transfer" CLOB);',
                share=True,
            ),
        ]

    @pytest.mark.slow
    @pytest.mark.database
    @pytest.mark.usefixtures(
        "use_globalworker_database",
        "create_state_table",
    )
    def test_udf_with_db(
        self,
        execute_udf_queries_in_db,
        globalworker_db_cursor,
    ):
        db = globalworker_db_cursor

        secure_transfer_, *_ = db.execute(
            "SELECT secure_transfer FROM __main"
        ).fetchone()
        result = json.loads(secure_transfer_)

        assert result == {
            "sum": {"data": 5, "operation": "sum", "type": "int"},
            "min": {"data": 5, "operation": "min", "type": "int"},
            "max": {"data": 5, "operation": "max", "type": "int"},
        }

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_SecureTransferOutput_with_SMPC_on(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            state=state(),
            return_type=secure_transfer(sum_op=True, max_op=True),
        )
        def f(state):
            result = {
                "sum": {"data": state["num"], "operation": "sum", "type": "int"},
                "max": {"data": state["num"], "operation": "max", "type": "int"},
            }
            return result

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
__udf()
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
    result = {'sum': {'data': state['num'], 'operation': 'sum', 'type': 'int'},
        'max': {'data': state['num'], 'operation': 'max', 'type': 'int'}}
    template, sum_op, min_op, max_op = udfio.split_secure_transfer_dict(result)
    _conn.execute(f"INSERT INTO __mainsum VALUES ('{json.dumps(sum_op)}');")
    _conn.execute(f"INSERT INTO __mainmax VALUES ('{json.dumps(max_op)}');")
    return json.dumps(template)
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenSMPCResult(
                template=UDFGenTableResult(
                    table_name="__main",
                    table_schema=[
                        ("secure_transfer", DType.JSON),
                    ],
                    create_query='CREATE TABLE __main("secure_transfer" CLOB);',
                    share=True,
                ),
                sum_op_values=UDFGenTableResult(
                    table_name="__mainsum",
                    table_schema=[
                        ("secure_transfer", DType.JSON),
                    ],
                    create_query='CREATE TABLE __mainsum("secure_transfer" CLOB);',
                ),
                max_op_values=UDFGenTableResult(
                    table_name="__mainmax",
                    table_schema=[
                        ("secure_transfer", DType.JSON),
                    ],
                    create_query='CREATE TABLE __mainmax("secure_transfer" CLOB);',
                ),
            )
        ]

    @pytest.fixture(scope="class")
    @pytest.mark.slow
    @pytest.mark.database
    @pytest.mark.usefixtures(
        "use_globalworker_database",
        "create_state_table",
    )
    def test_udf_with_db(
        self,
        execute_udf_queries_in_db,
        globalworker_db_cursor,
    ):
        db = globalworker_db_cursor

        template_str, *_ = db.execute("SELECT secure_transfer FROM __main").fetchone()
        template = json.loads(template_str)

        assert template == {
            "max": {"data": 0, "operation": "max", "type": "int"},
            "sum": {"data": 0, "operation": "sum", "type": "int"},
        }

        sum_op_values_str, *_ = db.execute(
            "SELECT secure_transfer FROM __main_sum_op"
        ).fetchone()
        sum_op_values = json.loads(sum_op_values_str)

        assert sum_op_values == [5]

        max_op_values_str, *_ = db.execute(
            "SELECT secure_transfer FROM __main_max_op"
        ).fetchone()
        max_op_values = json.loads(max_op_values_str)

        assert max_op_values == [5]

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
            smpc_used=True,
        )
        definition = gen.get_definition(udf_name="__udf", output_table_names=["__main"])
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_SecureTransferOutputAs2ndOutput_with_SMPC_off(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            state=state(),
            return_type=[
                state(),
                secure_transfer(sum_op=True, min_op=True, max_op=True),
            ],
        )
        def f(state):
            result = {
                "sum": {"data": state["num"], "operation": "sum", "type": "int"},
                "min": {"data": state["num"], "operation": "min", "type": "int"},
                "max": {"data": state["num"], "operation": "max", "type": "int"},
            }
            return state, result

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
__udf()
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
    result = {'sum': {'data': state['num'], 'operation': 'sum', 'type': 'int'},
        'min': {'data': state['num'], 'operation': 'min', 'type': 'int'}, 'max':
        {'data': state['num'], 'operation': 'max', 'type': 'int'}}
    _conn.execute(f"INSERT INTO __lt0 VALUES ('{json.dumps(result)}');")
    return pickle.dumps(state)
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("state", DType.BINARY),
                ],
                create_query='CREATE TABLE __main("state" BLOB);',
            ),
            UDFGenTableResult(
                table_name="__lt0",
                table_schema=[
                    ("secure_transfer", DType.JSON),
                ],
                create_query='CREATE TABLE __lt0("secure_transfer" CLOB);',
                share=True,
            ),
        ]

    @pytest.mark.slow
    @pytest.mark.database
    @pytest.mark.usefixtures(
        "use_globalworker_database",
        "create_state_table",
    )
    def test_udf_with_db(
        self,
        execute_udf_queries_in_db,
        globalworker_db_cursor,
    ):
        db = globalworker_db_cursor

        secure_transfer_, *_ = db.execute(
            "SELECT secure_transfer FROM __lt0"
        ).fetchone()
        result = json.loads(secure_transfer_)

        assert result == {
            "sum": {"data": 5, "operation": "sum", "type": "int"},
            "min": {"data": 5, "operation": "min", "type": "int"},
            "max": {"data": 5, "operation": "max", "type": "int"},
        }

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(
            udf_name="__udf", output_table_names=["__main", "__lt0"]
        )
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(
            udf_name="__udf", output_table_names=["__main", "__lt0"]
        )
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main", "__lt0"])
        assert len(results) == 2
        assert results[0] == expected_udf_outputs[0]
        assert results[1] == expected_udf_outputs[1]


class TestUDFGen_SecureTransferOutputAs2ndOutput_with_SMPC_on(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            state=state(),
            return_type=[
                state(),
                secure_transfer(sum_op=True, min_op=True, max_op=True),
            ],
        )
        def f(state):
            result = {
                "sum": {"data": state["num"], "operation": "sum", "type": "int"},
                "min": {"data": state["num"], "operation": "min", "type": "int"},
                "max": {"data": state["num"], "operation": "max", "type": "int"},
            }
            return state, result

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
__udf()
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
    result = {'sum': {'data': state['num'], 'operation': 'sum', 'type': 'int'},
        'min': {'data': state['num'], 'operation': 'min', 'type': 'int'}, 'max':
        {'data': state['num'], 'operation': 'max', 'type': 'int'}}
    template, sum_op, min_op, max_op = udfio.split_secure_transfer_dict(result)
    _conn.execute(f"INSERT INTO __lt0 VALUES ('{json.dumps(template)}');")
    _conn.execute(f"INSERT INTO __lt0sum VALUES ('{json.dumps(sum_op)}');")
    _conn.execute(f"INSERT INTO __lt0min VALUES ('{json.dumps(min_op)}');")
    _conn.execute(f"INSERT INTO __lt0max VALUES ('{json.dumps(max_op)}');")
    return pickle.dumps(state)
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("state", DType.BINARY),
                ],
                create_query='CREATE TABLE __main("state" BLOB);',
            ),
            UDFGenSMPCResult(
                template=UDFGenTableResult(
                    table_name="__lt0",
                    table_schema=[
                        ("secure_transfer", DType.JSON),
                    ],
                    create_query='CREATE TABLE __lt0("secure_transfer" CLOB);',
                    share=True,
                ),
                sum_op_values=UDFGenTableResult(
                    table_name="__lt0sum",
                    table_schema=[
                        ("secure_transfer", DType.JSON),
                    ],
                    create_query='CREATE TABLE __lt0sum("secure_transfer" CLOB);',
                ),
                min_op_values=UDFGenTableResult(
                    table_name="__lt0min",
                    table_schema=[
                        ("secure_transfer", DType.JSON),
                    ],
                    create_query='CREATE TABLE __lt0min("secure_transfer" CLOB);',
                ),
                max_op_values=UDFGenTableResult(
                    table_name="__lt0max",
                    table_schema=[
                        ("secure_transfer", DType.JSON),
                    ],
                    create_query='CREATE TABLE __lt0max("secure_transfer" CLOB);',
                ),
            ),
        ]

    @pytest.mark.slow
    @pytest.mark.database
    @pytest.mark.usefixtures(
        "use_globalworker_database",
        "create_state_table",
    )
    def test_udf_with_db(
        self,
        execute_udf_queries_in_db,
        globalworker_db_cursor,
    ):
        db = globalworker_db_cursor

        template_str, *_ = db.execute("SELECT secure_transfer FROM __lt0").fetchone()
        template = json.loads(template_str)
        assert template == {
            "sum": {"data": 0, "operation": "sum", "type": "int"},
            "min": {"data": 0, "operation": "min", "type": "int"},
            "max": {"data": 0, "operation": "max", "type": "int"},
        }

        sum_op_values_str, *_ = db.execute(
            "SELECT secure_transfer FROM __lt0sum"
        ).fetchone()
        sum_op_values = json.loads(sum_op_values_str)
        assert sum_op_values == [5]

        min_op_values_str, *_ = db.execute(
            "SELECT secure_transfer FROM __lt0min"
        ).fetchone()
        min_op_values = json.loads(min_op_values_str)
        assert min_op_values == [5]

        max_op_values_str, *_ = db.execute(
            "SELECT secure_transfer FROM __lt0max"
        ).fetchone()
        max_op_values = json.loads(max_op_values_str)
        assert max_op_values == [5]

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
            smpc_used=True,
        )
        definition = gen.get_definition(
            udf_name="__udf", output_table_names=["__main", "__lt0"]
        )
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(
            udf_name="__udf", output_table_names=["__main", "__lt0"]
        )
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main", "__lt0"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]
        assert results[1] == expected_udf_outputs[1]


class TestUDFGen_SecureTransferInput_with_SMPC_off(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            transfer=secure_transfer(sum_op=True),
            return_type=transfer(),
        )
        def f(transfer):
            return transfer

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
__udf()
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
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("transfer", DType.JSON),
                ],
                create_query='CREATE TABLE __main("transfer" CLOB);',
                share=True,
            ),
        ]

    @pytest.mark.slow
    @pytest.mark.database
    @pytest.mark.usefixtures(
        "use_globalworker_database",
        "create_secure_transfer_table",
    )
    def test_udf_with_db(
        self,
        execute_udf_queries_in_db,
        globalworker_db_cursor,
    ):
        db = globalworker_db_cursor

        transfer, *_ = db.execute("SELECT transfer FROM __main").fetchone()
        result = json.loads(transfer)

        assert result == {"sum": 111}

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_SecureTransferInput_with_SMPC_on(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            transfer=secure_transfer(sum_op=True, max_op=True),
            return_type=transfer(),
        )
        def f(transfer):
            return transfer

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
                sum_op=TableInfo(
                    name="test_smpc_sum_op_values_table",
                    schema_=TableSchema(
                        columns=[
                            ColumnInfo(name="secure_transfer", dtype=DType.JSON),
                        ]
                    ),
                    type_=TableType.NORMAL,
                ),
                max_op=TableInfo(
                    name="test_smpc_max_op_values_table",
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
__udf()
RETURNS
TABLE("transfer" CLOB)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    import json
    __template_str = _conn.execute("SELECT secure_transfer from test_smpc_template_table;")["secure_transfer"][0]
    __template = json.loads(__template_str)
    __sum_op_values_str = _conn.execute("SELECT secure_transfer from test_smpc_sum_op_values_table;")["secure_transfer"][0]
    __sum_op_values = json.loads(__sum_op_values_str)
    __min_op_values = None
    __max_op_values_str = _conn.execute("SELECT secure_transfer from test_smpc_max_op_values_table;")["secure_transfer"][0]
    __max_op_values = json.loads(__max_op_values_str)
    transfer = udfio.construct_secure_transfer_dict(__template,__sum_op_values,__min_op_values,__max_op_values)
    return json.dumps(transfer)
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("transfer", DType.JSON),
                ],
                create_query='CREATE TABLE __main("transfer" CLOB);',
                share=True,
            ),
        ]

    @pytest.mark.slow
    @pytest.mark.database
    @pytest.mark.usefixtures(
        "use_globalworker_database",
        "create_smpc_template_table_with_sum_and_max",
        "create_smpc_sum_op_values_table",
        "create_smpc_max_op_values_table",
    )
    def test_udf_with_db(
        self,
        execute_udf_queries_in_db,
        globalworker_db_cursor,
    ):
        db = globalworker_db_cursor

        transfer, *_ = db.execute("SELECT transfer FROM __main").fetchone()
        result = json.loads(transfer)

        assert result == {"sum": [100, 200, 300], "max": 58}

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
            smpc_used=True,
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_LoggerArgument(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(
            t=literal(),
            logger=udf_logger(),
            return_type=transfer(),
        )
        def f(t, logger):
            logger.info("Log inside monetdb udf.")
            result = {"num": t}
            return result

    @pytest.fixture(scope="class")
    def positional_args(self):
        return [5]

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
__udf()
RETURNS
TABLE("transfer" CLOB)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    import json
    t = 5
    logger = udfio.get_logger('__udf', '123')
    logger.info('Log inside monetdb udf.')
    result = {'num': t}
    return json.dumps(result)
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("transfer", DType.JSON),
                ],
                create_query='CREATE TABLE __main("transfer" CLOB);',
                share=True,
            )
        ]

    @pytest.mark.slow
    @pytest.mark.database
    @pytest.mark.usefixtures("use_globalworker_database")
    def test_udf_with_db(
        self,
        execute_udf_queries_in_db,
        globalworker_db_cursor,
    ):
        db = globalworker_db_cursor

        [transfer] = db.execute("SELECT * FROM __main").fetchone()
        result = json.loads(transfer)

        assert result == {"num": 5}

    def test_generate_udf_queries(
        self,
        funcname,
        positional_args,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=positional_args,
            flowkwargs={},
            request_id="123",
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_DeferredOutputSchema(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(return_type=relation(schema=DEFERRED))
        def f():
            result = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}
            return result

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
__udf()
RETURNS
TABLE("a" INT,"b" DOUBLE)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    result = {'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]}
    return udfio.as_relational_table(result, 'row_id')
}"""

    @pytest.fixture(scope="class")
    def expected_udfexec(self):
        return """\
INSERT INTO __main
SELECT
    *
FROM
    __udf();"""

    @pytest.fixture(scope="class")
    def expected_udf_outputs(self):
        return [
            UDFGenTableResult(
                table_name="__main",
                table_schema=[
                    ("a", DType.INT),
                    ("b", DType.FLOAT),
                ],
                create_query='CREATE TABLE __main("a" INT,"b" DOUBLE);',
            )
        ]

    def test_generate_udf_queries(
        self,
        funcname,
        expected_udfdef,
        expected_udfexec,
        expected_udf_outputs,
    ):
        output_schema = [("a", DType.INT), ("b", DType.FLOAT)]
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=[],
            flowkwargs={},
            output_schema=output_schema,
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
        exec = gen.get_exec_stmt(udf_name="__udf", output_table_names=["__main"])
        assert exec == expected_udfexec
        results = gen.get_results(output_table_names=["__main"])
        assert len(results) == len(expected_udf_outputs)
        assert results[0] == expected_udf_outputs[0]


class TestUDFGen_MinRowCountInputType(TestUDFGenBase):
    def define_pyfunc(self):
        @udf(a=MIN_ROW_COUNT, return_type=transfer())
        def f(a):
            result = {"a": a}
            return result

    @pytest.fixture(scope="class")
    def expected_udfdef(self):
        return """\
CREATE OR REPLACE FUNCTION
__udf()
RETURNS
TABLE("transfer" CLOB)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    import json
    a = 10
    result = {'a': a}
    return json.dumps(result)
}"""

    def test_generate_udf_queries(
        self,
        funcname,
        expected_udfdef,
    ):
        gen = PyUdfGenerator(
            udf.registry,
            func_name=funcname,
            flowargs=[],
            flowkwargs={},
            min_row_count=10,
        )
        definition = gen.get_definition(udf_name="__udf")
        assert definition == expected_udfdef
