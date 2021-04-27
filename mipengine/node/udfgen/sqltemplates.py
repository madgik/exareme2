from textwrap import indent
from enum import Enum
from mipengine.node.udfgen.udfgenerator2 import TensorArg


class BinaryOp(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MATMUL = "@"


NODEID_TEMPLATE = "{nodeid_column}"

SELECT = "SELECT"
FROM = "FROM"
WHERE = "WHERE"
GROUP_BY = "GROUP BY"
ORDER_BY = "ORDER BY"

SPC4 = 4 * " "
SEP = ","
LN = "\n"
SEPLN = SEP + LN


def get_tensor_binary_op_template(
    tensor_0: TensorArg,
    tensor_1: TensorArg,
    operator: BinaryOp,
):
    if operator is BinaryOp.MATMUL:
        return get_tensor_matmul_template(tensor_0, tensor_1)
    return get_tensor_elementwise_binary_op_template(tensor_0, tensor_1, operator)


def get_tensor_elementwise_binary_op_template(
    tensor_0: TensorArg,
    tensor_1: TensorArg,
    operator: BinaryOp,
):
    if tensor_0.ndims != tensor_1.ndims:
        raise NotImplementedError(
            "Cannot perform elementwise operation if the operand "
            f"dimensions are different: {tensor_0.ndims}, {tensor_1.ndims}"
        )
    ndims = tensor_0.ndims
    op = operator.value

    select_lines = get_select_lines_for_tensor_elementwise_binop(ndims, op)
    from_lines = get_crossproduct_from_lines(tensor_0, tensor_1)
    where_lines = get_where_lines_for_tensor_elementwise_binop(ndims)
    lines = [SELECT, select_lines, FROM, from_lines, WHERE, where_lines]
    return LN.join(lines)


def get_where_lines_for_tensor_elementwise_binop(ndims):
    where_lines = [f"tensor_0.dim{i}=tensor_1.dim{i}" for i in range(ndims)]
    return format_inner_query_lines(where_lines)


def get_crossproduct_from_lines(tensor_0, tensor_1):
    from_lines = [
        f"{tensor_0.table_name} AS tensor_0",
        f"{tensor_1.table_name} AS tensor_1",
    ]
    return format_inner_query_lines(from_lines)


def get_select_lines_for_tensor_elementwise_binop(ndims, op):
    select_lines = [
        NODEID_TEMPLATE,
        *[f"tensor_0.dim{i} AS dim{i}" for i in range(ndims)],
        f"tensor_0.val {op} tensor_1.val AS val",
    ]
    return format_inner_query_lines(select_lines)


def get_tensor_matmul_template(tensor_0: TensorArg, tensor_1: TensorArg):
    ndims0, ndims1 = tensor_0.ndims, tensor_1.ndims
    if ndims0 not in (1, 2) or ndims1 not in (1, 2):
        raise NotImplementedError(
            "Cannot multiply tensors of dimension greated than 2."
        )

    select_lines = get_select_lines_for_tensor_matmul(ndims0, ndims1)
    from_lines = get_crossproduct_from_lines(tensor_0, tensor_1)
    where_lines = get_where_lines_for_tensor_matmul(ndims0)
    groupby_lines = get_groupby_lines_for_tensor_matmul(ndims0, ndims1)
    orderby_lines = get_orderby_lines_for_tensor_matmul(ndims0, ndims1)

    lines = [SELECT, select_lines, FROM, from_lines, WHERE, where_lines]
    if groupby_lines:
        lines += [GROUP_BY, groupby_lines]
    if orderby_lines:
        lines += [ORDER_BY, orderby_lines]
    return LN.join(lines)


def get_select_lines_for_tensor_matmul(ndims0, ndims1):
    """After a contraction, the resulting tensor will contain all indices
    whicha are not contractes. The select part select all those indices."""
    select_lines = [NODEID_TEMPLATE]
    remaining_dims = compute_remaining_dimensions_after_contraction(ndims0, ndims1)

    # The (1, 2) case is an exception to the rule, where a transposition is
    # also required on the result. This is due to the fact that vec @ mat is,
    # strictly speaking, actually vec.T @ mat. Numpy, however, allows the
    # operation without requiring a transposition on the first operand and I
    # try to follow numpy's behaviour as much as possible.
    if (ndims0, ndims1) == (1, 2):
        select_lines += ["tensor_1.dim1 AS dim0"]
    else:
        select_lines += [f"tensor_{i}.dim{i} AS dim{i}" for i in range(remaining_dims)]

    select_lines += ["SUM(tensor_0.val * tensor_1.val) AS val"]
    return format_inner_query_lines(select_lines)


def get_where_lines_for_tensor_matmul(ndims0):
    """The where clause enforces equality for contracted indices."""
    ci0, ci1 = compute_contracted_indices(ndims0)
    where_lines = [f"tensor_0.dim{ci0}=tensor_1.dim{ci1}"]
    return format_inner_query_lines(where_lines)


def get_groupby_lines_for_tensor_matmul(ndims0, ndims1):
    """Similar to the select case, indices which are not contracted are grouped
    together in order to compute the sum of products over all combinations of
    non contracted indices."""
    ri0, ri1 = compute_remaining_indices_after_contraction(ndims0, ndims1)
    groupby_lines = [f"tensor_0.dim{i}" for i in ri0] + [
        f"tensor_1.dim{i}" for i in ri1
    ]
    return format_inner_query_lines(groupby_lines)


def get_orderby_lines_for_tensor_matmul(ndims0, ndims1):
    """The order by clause is not required for correctness but for readability
    of the result. All indices in the resulting table are ordered by ascending
    dimension."""
    remaining_dims = compute_remaining_dimensions_after_contraction(ndims0, ndims1)
    orderby_lines = [f"dim{i}" for i in range(remaining_dims)]
    return format_inner_query_lines(orderby_lines)


def compute_contracted_indices(ndims0):
    """During matrix multiplication the last index of the first tensor gets
    contracted with the first index of the second tensor."""
    return ndims0 - 1, 0


def compute_remaining_dimensions_after_contraction(ndims0, ndims1):
    return (ndims0 - 1) + (ndims1 - 1)


def compute_remaining_indices_after_contraction(ndims0, ndims1):
    *indices0, _ = range(ndims0)
    _, *indices1 = range(ndims1)
    return indices0, indices1


def format_inner_query_lines(lines):
    return SEPLN.join(indent(line, SPC4) for line in lines)


# ~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

import pytest

from mipengine.node.udfgen.udfgenerator2 import tensor, UDFBadCall


def test_tensor_elementwise_binary_op_1dim():
    tensor_0 = TensorArg(table_name="tens0", dtype=None, ndims=1)
    tensor_1 = TensorArg(table_name="tens1", dtype=None, ndims=1)
    op = BinaryOp.ADD
    expected = """\
SELECT
    {nodeid_column},
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
    op = BinaryOp.ADD
    expected = """\
SELECT
    {nodeid_column},
    tensor_0.dim0 AS dim0,
    tensor_0.dim1 AS dim1,
    tensor_0.val + tensor_1.val AS val
FROM
    tens0 AS tensor_0,
    tens1 AS tensor_1
WHERE
    tensor_0.dim0=tensor_1.dim0,
    tensor_0.dim1=tensor_1.dim1"""
    result = get_tensor_binary_op_template(tensor_0, tensor_1, op)
    assert result == expected


def test_vector_dot_vector_template():
    tensor_0 = TensorArg(table_name="vec0", dtype=None, ndims=1)
    tensor_1 = TensorArg(table_name="vec1", dtype=None, ndims=1)
    op = BinaryOp.MATMUL
    expected = """\
SELECT
    {nodeid_column},
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
    op = BinaryOp.MATMUL
    expected = """\
SELECT
    {nodeid_column},
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
    op = BinaryOp.MATMUL
    expected = """\
SELECT
    {nodeid_column},
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
    op = BinaryOp.MATMUL
    expected = """\
SELECT
    {nodeid_column},
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
