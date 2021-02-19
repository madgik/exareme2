from textwrap import dedent

from mipengine.node.udfgen.udfgenerator import generate_udf
from mipengine.node.udfgen.udfgenerator import monet_udf
from mipengine.node.udfgen.udfparams import Table
from mipengine.node.udfgen.udfparams import Tensor
from mipengine.node.udfgen.udfparams import LoopbackTable
from mipengine.node.udfgen.udfparams import LiteralParameter
import pytest


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


test_cases = [
    (
        udf_tables.to_sql(Table(int, (10, 2)), Table(int, (10, 2))),
        dedent(
            """
        CREATE OR REPLACE
        FUNCTION
        udf_tables(x0 BIGINT, x1 BIGINT, y0 BIGINT, y1 BIGINT)
        RETURNS
        Table(result0 BIGINT, result1 BIGINT)
        LANGUAGE PYTHON
        {
            x = ArrayBundle(_columns[0:2])
            y = ArrayBundle(_columns[2:4])

            # body
            result = x + y

            return as_relational_table(result)
        };"""[1:]
        ),
    ),
    (
        udf_table_to_tensor.to_sql(Table(int, (4, 4))),
        dedent(
            """
        CREATE OR REPLACE
        FUNCTION
        udf_table_to_tensor(input0 BIGINT, input1 BIGINT, input2 BIGINT, input3 BIGINT)
        RETURNS
        Table(input0 BIGINT, input1 BIGINT, input2 BIGINT, input3 BIGINT)
        LANGUAGE PYTHON
        {
            input = ArrayBundle(_columns[0:4])
            return as_relational_table(input)
        };"""[1:]
        ),
    ),
    (
        udf_tensor_and_loopback.to_sql(
            Tensor(float, (10, 3)),
            LoopbackTable(dtype=float, shape=(3, 1), name="coeffs"),
        ),
        dedent(
            """
        CREATE OR REPLACE
        FUNCTION
        udf_tensor_and_loopback(X0 DOUBLE, X1 DOUBLE, X2 DOUBLE)
        RETURNS
        Table(result0 DOUBLE)
        LANGUAGE PYTHON
        {
            X = from_tensor_table(_columns[0:3])
            c = _conn.execute("SELECT * FROM coeffs")

            # body
            result = X @ c

            return as_relational_table(result)
        };"""[1:]
        ),
    ),
    (
        udf_with_literals.to_sql(Tensor(int, (3, 3)), LiteralParameter(5)),
        dedent(
            """
            CREATE OR REPLACE
            FUNCTION
            udf_with_literals(X0 BIGINT, X1 BIGINT, X2 BIGINT)
            RETURNS
            Table(result0 BIGINT, result1 BIGINT, result2 BIGINT)
            LANGUAGE PYTHON
            {
                X = from_tensor_table(_columns[0:3])
                val = 5

                # body
                result = X + val

                return as_relational_table(result)
            };"""[1:]
        ),
    ),
]


@pytest.mark.parametrize("test_input,expected", test_cases)
def test_two_tables_in_table_out(test_input, expected):
    assert test_input == expected
