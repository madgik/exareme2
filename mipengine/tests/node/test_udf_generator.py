from textwrap import dedent

import pytest

from mipengine.algorithms import udf
from mipengine.algorithms import TableT
from mipengine.algorithms import TensorT
from mipengine.algorithms import LoopbackTableT
from mipengine.algorithms import LiteralParameterT
from mipengine.algorithms import ScalarT
from mipengine.node.udfgen.udfgenerator import generate_udf
from mipengine.node.udfgen.udfparams import Table
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
def udf_multitype_input(x: TableT, y: TensorT, z: LoopbackTableT) -> TableT:
    t = y * z
    return x


@udf
def udf_to_scalar(vec1: TensorT, vec2: TensorT) -> ScalarT:
    dotprod = vec1.T @ vec2
    return dotprod


test_cases_generate_udf = [
    (
        {
            "func_name": "udf_tables",
            "udf_name": "udf_tables_1234",
            "input_tables": [
                {"schema": [{"type": int}, {"type": int}], "nrows": 10},
                {"schema": [{"type": int}, {"type": int}], "nrows": 10},
            ],
            "loopback_tables": [],
            "literalparams": {},
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
            "func_name": "udf_table_to_tensor",
            "udf_name": "udf_table_to_tensor_1234",
            "input_tables": [
                {
                    "schema": [
                        {"type": int},
                        {"type": int},
                        {"type": int},
                        {"type": int},
                    ],
                    "nrows": 4,
                }
            ],
            "loopback_tables": [],
            "literalparams": {},
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
            "func_name": "udf_tensor_and_loopback",
            "udf_name": "udf_tensor_and_loopback_1234",
            "input_tables": [
                {
                    "schema": [
                        {"type": float},
                        {"type": float},
                        {"type": float},
                    ],
                    "nrows": 10,
                }
            ],
            "loopback_tables": [
                {"schema": [{"type": float}], "nrows": 3, "name": "coeffs"}
            ],
            "literalparams": {},
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
            "func_name": "udf_with_literals",
            "input_tables": [
                {"schema": [{"type": int}, {"type": int}, {"type": int}], "nrows": 3}
            ],
            "loopback_tables": [],
            "literalparams": {"val": 5},
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
            "func_name": "udf_multitype_input",
            "input_tables": [
                {"schema": [{"type": int}, {"type": int}], "nrows": 10},
                {"schema": [{"type": float}, {"type": float}], "nrows": 2},
            ],
            "loopback_tables": [
                {"schema": [{"type": float}, {"type": float}], "nrows": 2, "name": "t"}
            ],
            "literalparams": {},
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
                y = from_tensor_table(_columns[2:4])
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
            "func_name": "udf_to_scalar",
            "input_tables": [
                {"schema": [{"type": int}], "nrows": 10},
                {"schema": [{"type": int}], "nrows": 10},
            ],
            "loopback_tables": [],
            "literalparams": {},
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
]


@pytest.mark.parametrize("test_input,expected", test_cases_generate_udf)
def test_generate_udf(test_input, expected):
    assert (
        generate_udf(
            func_name=test_input["func_name"],
            udf_name=test_input["udf_name"],
            input_tables=test_input["input_tables"],
            loopback_tables=test_input["loopback_tables"],
            literalparams=test_input["literalparams"],
        )
        == expected
    )
