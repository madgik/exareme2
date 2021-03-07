from textwrap import dedent

import pytest

from mipengine.algorithms import udf
from mipengine.algorithms import TableT
from mipengine.algorithms import TensorT
from mipengine.algorithms import LoopbackTableT
from mipengine.algorithms import LiteralParameterT
from mipengine.algorithms import ScalarT
from mipengine.node.udfgen import generate_udf
from mipengine.node.udfgen.udfparams import DatalessArray
from mipengine.node.udfgen.udfparams import Tensor
from mipengine.node.udfgen.udfparams import LoopbackTable
from mipengine.node.udfgen.udfparams import LiteralParameter


@udf
def udf_tables(x: TableT, y: TableT) -> TableT:
    result = x + y
    return result


@udf
def udf_table_to_tensor(input: TableT) -> TensorT:
    return input


@udf
def udf_tensor_and_loopback(X: TensorT, c: LoopbackTableT) -> TensorT:
    result = X @ c
    return result


@udf
def udf_with_literals(X: TensorT, val: LiteralParameterT) -> TensorT:
    result = X + val
    return result


@udf
def udf_multitype_input(x: TableT, y: TableT, z: LoopbackTableT) -> TableT:
    t = y * z
    return x


@udf
def udf_to_scalar(vec1: TensorT, vec2: TensorT) -> ScalarT:
    dotprod = vec1.T @ vec2
    return dotprod


@udf
def udf_many_params(
    x: TensorT, y: TensorT, z: LoopbackTableT, w: LiteralParameterT, t: TensorT
) -> ScalarT:
    if x is not None:
        if len(y) != 0:
            result = w
    return result


test_cases_generate_udf = [
    (
        {
            "func_name": "test_udf_generator.udf_tables",
            "udf_name": "udf_tables_1234",
            "positional_args": [
                {
                    "type": "input_table",
                    "schema": [{"type": "int"}, {"type": "int"}],
                    "nrows": 10,
                },
                {
                    "type": "input_table",
                    "schema": [{"type": "int"}, {"type": "int"}],
                    "nrows": 10,
                },
            ],
            "keyword_args": {},
        },
        dedent(
            """\
            CREATE OR REPLACE
            FUNCTION
            udf_tables_1234(x0 BIGINT, x1 BIGINT, y0 BIGINT, y1 BIGINT)
            RETURNS
            Table(result0 BIGINT, result1 BIGINT)
            LANGUAGE PYTHON
            {
                from mipengine.udfgen import ArrayBundle
                x = ArrayBundle(_columns[0:2])
                y = ArrayBundle(_columns[2:4])

                # body
                result = x + y

                return as_relational_table(result)
            };"""
        ),
    ),
    (
        {
            "func_name": "test_udf_generator.udf_table_to_tensor",
            "udf_name": "udf_table_to_tensor_1234",
            "positional_args": [
                {
                    "type": "input_table",
                    "schema": [
                        {"type": "int"},
                        {"type": "int"},
                        {"type": "int"},
                        {"type": "int"},
                    ],
                    "nrows": 4,
                }
            ],
            "keyword_args": {},
        },
        dedent(
            """\
            CREATE OR REPLACE
            FUNCTION
            udf_table_to_tensor_1234(input0 BIGINT, input1 BIGINT, input2 BIGINT, input3 BIGINT)
            RETURNS
            Table(input0 BIGINT, input1 BIGINT, input2 BIGINT, input3 BIGINT)
            LANGUAGE PYTHON
            {
                from mipengine.udfgen import ArrayBundle
                input = ArrayBundle(_columns[0:4])
                return as_tensor_table(input)
            };"""
        ),
    ),
    (
        {
            "func_name": "test_udf_generator.udf_tensor_and_loopback",
            "udf_name": "udf_tensor_and_loopback_1234",
            "positional_args": [
                {
                    "type": "input_table",
                    "schema": [
                        {"type": "real"},
                        {"type": "real"},
                        {"type": "real"},
                    ],
                    "nrows": 10,
                },
                {
                    "type": "loopback_table",
                    "schema": [{"type": "real"}],
                    "nrows": 3,
                    "name": "coeffs",
                },
            ],
            "keyword_args": {},
        },
        dedent(
            """\
            CREATE OR REPLACE
            FUNCTION
            udf_tensor_and_loopback_1234(X0 DOUBLE, X1 DOUBLE, X2 DOUBLE)
            RETURNS
            Table(result0 DOUBLE)
            LANGUAGE PYTHON
            {
                from mipengine.udfgen import ArrayBundle
                X = from_tensor_table(_columns[0:3])
                c = _conn.execute("SELECT * FROM coeffs")

                # body
                result = X @ c

                return as_tensor_table(result)
            };"""
        ),
    ),
    (
        {
            "udf_name": "udf_with_literals_1234",
            "func_name": "test_udf_generator.udf_with_literals",
            "positional_args": [
                {
                    "type": "input_table",
                    "schema": [{"type": "int"}, {"type": "int"}, {"type": "int"}],
                    "nrows": 3,
                },
                {"type": "literal_parameter", "value": 5},
            ],
            "keyword_args": {},
        },
        dedent(
            """\
            CREATE OR REPLACE
            FUNCTION
            udf_with_literals_1234(X0 BIGINT, X1 BIGINT, X2 BIGINT)
            RETURNS
            Table(result0 BIGINT, result1 BIGINT, result2 BIGINT)
            LANGUAGE PYTHON
            {
                from mipengine.udfgen import ArrayBundle
                X = from_tensor_table(_columns[0:3])
                val = 5

                # body
                result = X + val

                return as_tensor_table(result)
            };"""
        ),
    ),
    (
        {
            "udf_name": "udf_multitype_input_1234",
            "func_name": "test_udf_generator.udf_multitype_input",
            "positional_args": [
                {
                    "type": "input_table",
                    "schema": [{"type": "int"}, {"type": "int"}],
                    "nrows": 10,
                },
                {
                    "type": "input_table",
                    "schema": [{"type": "real"}, {"type": "real"}],
                    "nrows": 2,
                },
                {
                    "type": "loopback_table",
                    "schema": [{"type": "real"}, {"type": "real"}],
                    "nrows": 2,
                    "name": "t",
                },
            ],
            "keyword_args": {},
        },
        dedent(
            """\
            CREATE OR REPLACE
            FUNCTION
            udf_multitype_input_1234(x0 BIGINT, x1 BIGINT, y0 DOUBLE, y1 DOUBLE)
            RETURNS
            Table(x0 BIGINT, x1 BIGINT)
            LANGUAGE PYTHON
            {
                from mipengine.udfgen import ArrayBundle
                x = ArrayBundle(_columns[0:2])
                y = ArrayBundle(_columns[2:4])
                z = _conn.execute("SELECT * FROM t")

                # body
                t = y * z

                return as_relational_table(x)
            };"""
        ),
    ),
    (
        {
            "udf_name": "udf_to_scalar_1234",
            "func_name": "test_udf_generator.udf_to_scalar",
            "positional_args": [
                {"type": "input_table", "schema": [{"type": "int"}], "nrows": 10},
                {"type": "input_table", "schema": [{"type": "int"}], "nrows": 10},
            ],
            "keyword_args": {},
        },
        dedent(
            """\
                CREATE OR REPLACE
                FUNCTION
                udf_to_scalar_1234(vec10 BIGINT, vec20 BIGINT)
                RETURNS
                BIGINT
                LANGUAGE PYTHON
                {
                    from mipengine.udfgen import ArrayBundle
                    vec1 = from_tensor_table(_columns[0:1])
                    vec2 = from_tensor_table(_columns[1:2])

                    # body
                    dotprod = vec1.T @ vec2

                    return dotprod

                };"""
        ),
    ),
    (
        {
            "udf_name": "many_params_1029384",
            "func_name": "test_udf_generator.udf_many_params",
            "positional_args": [
                {
                    "type": "input_table",
                    "schema": [{"type": "int"}, {"type": "int"}],
                    "nrows": 10,
                },
            ],
            "keyword_args": {
                "y": {
                    "type": "input_table",
                    "schema": [{"type": "int"}, {"type": "int"}],
                    "nrows": 10,
                },
                "z": {
                    "type": "loopback_table",
                    "schema": [{"type": "real"}],
                    "nrows": 3,
                    "name": "coeffs",
                },
                "w": {"type": "literal_parameter", "value": 5},
                "t": {
                    "type": "input_table",
                    "schema": [{"type": "real"}, {"type": "real"}],
                    "nrows": 10,
                },
            },
        },
        dedent(
            """\
            CREATE OR REPLACE
            FUNCTION
            many_params_1029384(x0 BIGINT, x1 BIGINT, y0 BIGINT, y1 BIGINT, t0 DOUBLE, t1 DOUBLE)
            RETURNS
            BIGINT
            LANGUAGE PYTHON
            {
                from mipengine.udfgen import ArrayBundle
                x = from_tensor_table(_columns[0:2])
                y = from_tensor_table(_columns[2:4])
                t = from_tensor_table(_columns[4:6])
                z = _conn.execute("SELECT * FROM coeffs")
                w = 5

                # body
                if x is not None:
                    if len(y) != 0:
                        result = w

                return result

            };"""
        ),
    ),
]


@pytest.mark.parametrize("test_input,expected", test_cases_generate_udf)
def test_generate_udf(test_input, expected):
    assert (
        generate_udf(
            func_name=test_input["func_name"],
            udf_name=test_input["udf_name"],
            positional_args=test_input["positional_args"],
            keyword_args=test_input["keyword_args"],
        )
        == expected
    )
