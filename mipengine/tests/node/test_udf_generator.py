# type: ignore
from textwrap import dedent
from typing import TypeVar

import pytest

from mipengine.algorithms import udf
from mipengine.algorithms import RelationT
from mipengine.algorithms import TensorT
from mipengine.algorithms import LoopbackTensorT
from mipengine.algorithms import LiteralParameterT
from mipengine.algorithms import ScalarT
from mipengine.node.udfgen import generate_udf_application_queries
from mipengine.node.udfgen import ColumnInfo, TableInfo

Schema1 = TypeVar("Schema1")
Schema2 = TypeVar("Schema2")
DT1 = TypeVar("DT1")
ND1 = TypeVar("ND1")
DT2 = TypeVar("DT2")
ND2 = TypeVar("ND2")

UDFNAME = "udfname"
TABLENAME = "tablename"
NODEID = "12345"


@udf
def relations_to_relation(
    x: RelationT[Schema1], y: RelationT[Schema2]
) -> RelationT[Schema1]:
    result = x
    return result


POSARGS_relations_to_relation = [
    TableInfo("rel1", [ColumnInfo("col1", "int"), ColumnInfo("col2", "float")]),
    TableInfo("rel2", [ColumnInfo("col1", "int"), ColumnInfo("col2", "float")]),
]
DEF_relations_to_relation = """\
CREATE OR REPLACE
FUNCTION
udfname(x_col1 int, x_col2 float, y_col1 int, y_col2 float)
RETURNS
TABLE(col1 int, col2 float)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    x = pd.DataFrame({n: _columns[n] for n in ['x_col1', 'x_col2']})
    y = pd.DataFrame({n: _columns[n] for n in ['y_col1', 'y_col2']})
    result = x
    return result
};"""
QUERY_relations_to_relation = """\
DROP TABLE IF EXISTS tablename;
CREATE TABLE tablename AS (
    SELECT CAST('12345' AS varchar(50)) AS node_id, *
    FROM
        udfname(
            (
                SELECT
                    rel1.col1, rel1.col2, rel2.col1, rel2.col2
                FROM
                    rel1, rel2
                WHERE
                    rel1.row_id=rel2.row_id
            )
        )
);"""


@udf
def table_to_tensor(input_: RelationT[Schema1]) -> TensorT(float, 2):
    output = [[1, 2, 3], [4, 5, 6]]
    return output


POSARGS_table_to_tensor = [
    TableInfo("rel1", [ColumnInfo("col1", "int"), ColumnInfo("col2", "float")]),
]
DEF_table_to_tensor = """\
CREATE OR REPLACE
FUNCTION
udfname(input__col1 int, input__col2 float)
RETURNS
TABLE(dim0 int, dim1 int, val float)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    input_ = pd.DataFrame({n: _columns[n] for n in ['input__col1', 'input__col2']})
    output = [[1, 2, 3], [4, 5, 6]]
    return udfio.as_tensor_table(numpy.array(output))
};"""
QUERY_table_to_tensor = """\
DROP TABLE IF EXISTS tablename;
CREATE TABLE tablename AS (
    SELECT CAST('12345' AS varchar(50)) AS node_id, *
    FROM
        udfname(
            (
                SELECT
                    rel1.col1, rel1.col2
                FROM
                    rel1
            )
        )
);"""


# FIXME loopbacks are not implemented
# @udf
# def tensor_and_loopback(
#     X: TensorT(float, 2), c: LoopbackRelationT(float, 1)
# ) -> TensorT(float, 1):
#     result = X @ c
#     return result


@udf
def with_literal(X: TensorT[DT1, ND1], n: LiteralParameterT[int]) -> TensorT[DT1, ND1]:
    result = X + n
    return result


POSARGS_with_literal = [
    TableInfo("tens1", [ColumnInfo("dim0", "int"), ColumnInfo("val", "float")]),
    5,
]
DEF_with_literal = """\
CREATE OR REPLACE
FUNCTION
udfname(tens1_dim0 int, tens1_val float)
RETURNS
TABLE(dim0 int, val float)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    X = udfio.from_tensor_table({n:_columns[n] for n in ['tens1_dim0', 'tens1_val']})
    n = 5
    result = X + n
    return udfio.as_tensor_table(numpy.array(result))
};"""
QUERY_with_literal = """\
DROP TABLE IF EXISTS tablename;
CREATE TABLE tablename AS (
    SELECT CAST('12345' AS varchar(50)) AS node_id, *
    FROM
        udfname(
            (
                SELECT
                    tens1.dim0, tens1.val
                FROM
                    tens1
            )
        )
);"""


@udf
def to_scalar(vec1: TensorT(float, 1), vec2: TensorT(float, 1)) -> ScalarT(float):
    dotprod = vec1.T @ vec2
    return dotprod


POSARGS_to_scalar = [
    TableInfo("tens1", [ColumnInfo("dim0", "int"), ColumnInfo("val", "float")]),
    TableInfo("tens2", [ColumnInfo("dim0", "int"), ColumnInfo("val", "float")]),
]
DEF_to_scalar = """\
CREATE OR REPLACE
FUNCTION
udfname(tens1_dim0 int, tens1_val float, tens2_dim0 int, tens2_val float)
RETURNS
float
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    vec1 = udfio.from_tensor_table({n:_columns[n] for n in ['tens1_dim0', 'tens1_val']})
    vec2 = udfio.from_tensor_table({n:_columns[n] for n in ['tens2_dim0', 'tens2_val']})
    dotprod = vec1.T @ vec2
    return dotprod
};"""
QUERY_to_scalar = """\
DROP TABLE IF EXISTS tablename;
CREATE TABLE tablename AS (
    SELECT
        udfname(tens1.dim0, tens1.val, tens2.dim0, tens2.val)
    FROM
        tens1, tens2
    WHERE
        tens1.dim0=tens2.dim0
);"""

test_cases_generate_udf = [
    (
        {
            "func_name": "test_udf_generator.relations_to_relation",
            "positional_args": POSARGS_relations_to_relation,
            "keyword_args": {},
        },
        (DEF_relations_to_relation, QUERY_relations_to_relation),
    ),
    (
        {
            "func_name": "test_udf_generator.table_to_tensor",
            "positional_args": POSARGS_table_to_tensor,
            "keyword_args": {},
        },
        (DEF_table_to_tensor, QUERY_table_to_tensor),
    ),
    (
        {
            "func_name": "test_udf_generator.with_literal",
            "positional_args": POSARGS_with_literal,
            "keyword_args": {},
        },
        (DEF_with_literal, QUERY_with_literal),
    ),
    (
        {
            "func_name": "test_udf_generator.to_scalar",
            "positional_args": POSARGS_to_scalar,
            "keyword_args": {},
        },
        (DEF_to_scalar, QUERY_to_scalar),
    ),
]


@pytest.mark.parametrize("test_input,expected", test_cases_generate_udf)
def test_generate_udf(test_input, expected):
    udf_def, udf_query = generate_udf_application_queries(**test_input)
    exp_def, exp_query = expected
    assert udf_def.substitute(udf_name=UDFNAME) == exp_def
    assert (
        udf_query.substitute(udf_name=UDFNAME, table_name=TABLENAME, node_id=NODEID)
        == exp_query
    )
