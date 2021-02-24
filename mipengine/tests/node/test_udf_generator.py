from textwrap import dedent

import pytest

from mipengine.algorithms.udfgen.udfgenerator import generate_udf
from mipengine.algorithms.udfgen.udfgenerator import monet_udf
from mipengine.algorithms.udfgen.udfparams import Table
from mipengine.algorithms.udfgen.udfparams import Tensor
from mipengine.algorithms.udfgen.udfparams import LoopbackTable
from mipengine.algorithms.udfgen.udfparams import LiteralParameter


@monet_udf
def udf_tables(x: Table, y: Table) -> Table:
    result = x + y
    return result


@monet_udf
def udf_table_to_tensor(input: Table) -> Tensor:
    return input


@monet_udf
def udf_tensor_and_loopback(X: Tensor, c: LoopbackTable) -> Tensor:
    result = X @ c
    return result


@monet_udf
def udf_with_literals(X: Tensor, val: LiteralParameter) -> Tensor:
    result = X + val
    return result


@monet_udf
def udf_multitype_input(x: Table, y: Tensor, z: LoopbackTable) -> Table:
    t = y * z
    return x


test_cases_to_sql = [
    (
        {
            "udf": udf_tables,
            "inputs": [Table(int, (10, 2)), Table(int, (10, 2))],
            "name": "udf_tables_1234",
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
            "udf": udf_table_to_tensor,
            "inputs": [Table(int, (4, 4))],
            "name": "udf_table_to_tensor_1234",
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
                input = ArrayBundle(_columns[0:4])
                return as_tensor_table(input)
            };"""
        ),
    ),
    (
        {
            "udf": udf_tensor_and_loopback,
            "inputs": [
                Tensor(float, (10, 3)),
                LoopbackTable(dtype=float, shape=(3, 1), name="coeffs"),
            ],
            "name": "udf_tensor_and_loopback_1234",
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
            "udf": udf_with_literals,
            "inputs": [Tensor(int, (3, 3)), LiteralParameter(5)],
            "name": "udf_with_literals_1234",
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
            "udf": udf_multitype_input,
            "inputs": [
                Table(int, (10, 2)),
                Tensor(float, (2, 2)),
                LoopbackTable(dtype=float, shape=(2, 2), name="t"),
            ],
            "name": "udf_multitype_input_1234",
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
                x = ArrayBundle(_columns[0:2])
                y = from_tensor_table(_columns[2:4])
                z = _conn.execute("SELECT * FROM t")

                # body
                t = y * z

                return as_relational_table(x)
            };"""
        ),
    ),
]


@pytest.mark.parametrize("test_input,expected", test_cases_to_sql)
def test_udf_generator_to_sql(test_input, expected):
    name = test_input["name"]
    udf = test_input["udf"]
    inputs = test_input["inputs"]
    assert udf.to_sql(name, *inputs) == expected


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
                x = ArrayBundle(_columns[0:2])
                y = from_tensor_table(_columns[2:4])
                z = _conn.execute("SELECT * FROM t")

                # body
                t = y * z

                return as_relational_table(x)
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
