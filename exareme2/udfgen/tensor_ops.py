"""
This module implements functions for generating SQL queries for basic tensor
operations

These are the four arithmetic operations (+ - * /), dot product (@) for 1 and 2
dimensional tensors and matrix transposition.

WARNING These operations are now deprecated and this module shouldn't be used as
it might be completely removed in the future. If you need to perform operations
on tensors you should use numpy within UDFs.
"""
import operator
import warnings
from enum import Enum

from exareme2.udfgen.ast import Column
from exareme2.udfgen.ast import ScalarFunction
from exareme2.udfgen.ast import Select
from exareme2.udfgen.ast import Table
from exareme2.udfgen.iotypes import LiteralArg
from exareme2.udfgen.iotypes import TensorArg

warnings.warn(
    "The tensor_ops module is deprecated and should not be imported.",
    DeprecationWarning,
    stacklevel=2,
)


class TensorBinaryOp(Enum):
    ADD = operator.add
    SUB = operator.sub
    MUL = operator.mul
    DIV = operator.truediv
    MATMUL = operator.matmul


class TensorUnaryOp(Enum):
    TRANSPOSE = 0


TENSOR_OP_NAMES = TensorBinaryOp.__members__.keys() | TensorUnaryOp.__members__.keys()


def get_sql_tensor_operation_select_query(udf_posargs, func_name):
    if func_name == TensorUnaryOp.TRANSPOSE.name:
        assert len(udf_posargs) == 1
        matrix, *_ = udf_posargs
        assert matrix.ndims == 2
        return get_matrix_transpose_template(matrix)
    if len(udf_posargs) == 2:
        operand1, operand2 = udf_posargs
        return get_tensor_binary_op_template(
            operand1, operand2, getattr(TensorBinaryOp, func_name)
        )
    raise NotImplementedError


def get_tensor_binary_op_template(
    operand_0: TensorArg,
    operand_1: TensorArg,
    operator: TensorBinaryOp,
):
    if operator is TensorBinaryOp.MATMUL:
        return get_tensor_matmul_template(operand_0, operand_1)
    return get_tensor_elementwise_binary_op_template(operand_0, operand_1, operator)


def get_tensor_elementwise_binary_op_template(
    operand_0: TensorArg,
    operand_1: TensorArg,
    operator: TensorBinaryOp,
):
    if isinstance(operand_0, TensorArg) and isinstance(operand_1, TensorArg):
        return get_tensor_tensor_elementwise_op_template(operand_0, operand_1, operator)
    if isinstance(operand_0, LiteralArg) ^ isinstance(operand_1, LiteralArg):
        return get_tensor_number_binary_op_template(operand_0, operand_1, operator)
    raise NotImplementedError


def get_tensor_tensor_elementwise_op_template(tensor0, tensor1, operator):
    if tensor0.ndims != tensor1.ndims:
        raise NotImplementedError(
            "Cannot perform elementwise operation if the operand "
            f"dimensions are different: {tensor0.ndims}, {tensor1.ndims}"
        )
    table0 = convert_table_arg_to_table_ast_node(tensor0, alias="tensor_0")
    table1 = convert_table_arg_to_table_ast_node(tensor1, alias="tensor_1")

    columns = get_columns_for_tensor_tensor_binary_op(table0, table1, operator)
    where = get_where_params_for_tensor_tensor_binary_op(table0, table1)

    select_stmt = Select(
        columns=columns,
        tables=[table0, table1],
        where=where,
    )
    return select_stmt.compile()


def get_columns_for_tensor_tensor_binary_op(table0, table1, operator):
    columns = [
        column for name, column in table0.columns.items() if name.startswith("dim")
    ]
    for column in columns:
        column.alias = column.name
    valcolumn = operator.value(table0.c["val"], table1.c["val"])
    valcolumn.alias = "val"
    columns += [valcolumn]
    return columns


def get_where_params_for_tensor_tensor_binary_op(table0, table1):
    where = [
        table0.c[colname] == table1.c[colname]
        for colname in table0.columns
        if colname.startswith("dim")
    ]
    return where


def get_tensor_number_binary_op_template(operand_0, operand_1, operator):
    if isinstance(operand_0, LiteralArg):
        number = operand_0.value
        table = convert_table_arg_to_table_ast_node(operand_1, alias="tensor_0")
        valcolumn = operator.value(number, table.c["val"])
        valcolumn.alias = "val"
    else:
        number = operand_1.value
        table = convert_table_arg_to_table_ast_node(operand_0, alias="tensor_0")
        valcolumn = operator.value(table.c["val"], number)
        valcolumn.alias = "val"
    columns = get_columns_for_tensor_number_binary_op(table, valcolumn)
    select_stmt = Select(columns, tables=[table])

    return select_stmt.compile()


def get_columns_for_tensor_number_binary_op(table, valcolumn):
    columns = [
        column for name, column in table.columns.items() if name.startswith("dim")
    ]
    for column in columns:
        column.alias = column.name
    columns += [valcolumn]
    return columns


def get_tensor_matmul_template(tensor0: TensorArg, tensor1: TensorArg):
    ndims0, ndims1 = tensor0.ndims, tensor1.ndims
    if ndims0 not in (1, 2) or ndims1 not in (1, 2):
        raise NotImplementedError(
            "Cannot multiply tensors of dimension greated than 2."
        )
    table0 = convert_table_arg_to_table_ast_node(tensor0)
    table1 = convert_table_arg_to_table_ast_node(tensor1)
    table0.alias, table1.alias = "tensor_0", "tensor_1"
    ndims0, ndims1 = tensor0.ndims, tensor1.ndims

    columns = get_columns_for_tensor_matmul(ndims0, ndims1, table0, table1)
    where = get_where_params_for_tensor_matmul(ndims0, table0, table1)
    groupby = get_groupby_params_for_tensor_matmul(ndims0, ndims1, table0, table1)
    orderby = get_orderby_params_for_tensor_matmul(ndims0, ndims1)

    tables = [table0, table1]
    select_stmt = Select(columns, tables, where, groupby, orderby)
    return select_stmt.compile()


def get_columns_for_tensor_matmul(ndims0, ndims1, table0, table1):
    """After a contraction, the resulting tensor will contain all indices
    which are not contracted. The select part involves all those indices."""
    tables = [table0, table1]
    remaining_dims = compute_remaining_dimensions_after_contraction(ndims0, ndims1)
    # The (1, 2) case is an exception to the rule, where a transposition is
    # also required on the result. This is due to the fact that vec @ mat is,
    # strictly speaking, actually vec.T @ mat. Numpy, however, allows the
    # operation without requiring a transposition on the first operand and I
    # try to follow numpy's behaviour as much as possible.
    if (ndims0, ndims1) == (1, 2):
        tables[1].c["dim1"].alias = "dim0"
        columns = [tables[1].c["dim1"]]
    else:
        for i in range(remaining_dims):
            tables[i].c[f"dim{i}"].alias = f"dim{i}"
        columns = [tables[i].c[f"dim{i}"] for i in range(remaining_dims)]

    prods_column = table0.c["val"] * table1.c["val"]
    sum_of_prods = ScalarFunction("SUM", [prods_column], alias="val")
    columns += [sum_of_prods]
    return columns


def get_where_params_for_tensor_matmul(ndims0, table_0, table_1):
    """The where clause enforces equality on contracted indices."""
    ci0, ci1 = compute_contracted_indices(ndims0)
    where_params = [table_0.c[f"dim{ci0}"] == table_1.c[f"dim{ci1}"]]
    return where_params


def get_groupby_params_for_tensor_matmul(ndims0, ndims1, table0, table1):
    """Similar to the select case, indices which are not contracted are grouped
    together in order to compute the sum of products over all combinations of
    non contracted indices."""
    ri0, ri1 = compute_remaining_indices_after_contraction(ndims0, ndims1)
    groupby_params = [table0.c[f"dim{i}"] for i in ri0]
    groupby_params += [table1.c[f"dim{i}"] for i in ri1]
    return groupby_params


def get_orderby_params_for_tensor_matmul(ndims0, ndims1):
    """The order by clause is not required for correctness but for readability.
    All indices in the resulting table are ordered by ascending dimension."""
    remaining_dims = compute_remaining_dimensions_after_contraction(ndims0, ndims1)
    orderby_params = [Column(f"dim{i}") for i in range(remaining_dims)]
    return orderby_params


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


def get_matrix_transpose_template(matrix):
    table = Table(
        name=matrix.table_name,
        columns=matrix.column_names(),
        alias="tensor_0",
    )
    table.c["dim0"].alias = "dim1"
    table.c["dim1"].alias = "dim0"
    table.c["val"].alias = "val"
    select_stmt = Select(
        [table.c["dim1"], table.c["dim0"], table.c["val"]],
        [table],
    )
    return select_stmt.compile()


def convert_table_arg_to_table_ast_node(table_arg, alias=None):
    return Table(
        name=table_arg.table_name,
        columns=table_arg.column_names(),
        alias=alias,
    )
