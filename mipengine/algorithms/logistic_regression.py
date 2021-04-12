# type: ignore
from numbers import Number
from typing import DefaultDict
from typing import TypeVar
from typing import Any

import numpy
import pandas as pd

from mipengine.algorithms import udf
from mipengine.algorithms import TableT
from mipengine.algorithms import TensorT
from mipengine.algorithms import LiteralParameterT
from mipengine.algorithms import ScalarT
from mipengine.algorithms import RelationT


PREC = 1e-6


def run(algo_interface):

    # y: TableT, X: TableT
    run_on_locals = algo_interface.run_udf_on_local_nodes
    run_on_global = algo_interface.run_udf_on_global_node
    get_table_schema = algo_interface.get_table_schema

    X_init: "LocalNodeTable" = algo_interface.initial_view_tables['x']
    X = run_on_locals(func_name="logistic_regression.relation_to_matrix", positional_args={"rel": X_init})  # X = relation_to_matrix(X)

    y_init: "LocalNodeTable" = algo_interface.initial_view_tables['y']
    y = run_on_locals(func_name="logistic_regression.relation_to_vector", positional_args={"rel": y_init})  # y = relation_to_vector(y)

    # init model
    table_schema = get_table_schema(X)  #nobs, ncols = X.shape
    ncols = len(table_schema.columns)

    coeff = run_on_locals(func_name="sql.zeros1", positional_args={"n": ncols})  # coeff = zeros1(ncols)


    logloss = 1e6
    # loop update coefficients
    while True:
        z = run_on_locals(func_name="sql.matrix_dot_vector", positional_args={"Μ": X, "v": coeff})  # z = matrix_dot_vector(X, coeff)

        s = run_on_locals(func_name="logistic_regression.tensor_expit", positional_args={"t": z})  # s = tensor_expit(z)

        tmp = run_on_locals(func_name="sql.const_tensor1_sub", positional_args={"const": 1, "t": s})  # tmp = const_tensor_sub(1, s)

        d = run_on_locals(func_name="sql.tensor1_mult", positional_args={"t1": s, "t2": tmp})  # d = tensor_mult(s, const_tensor_sub(1, s))

        tmp = run_on_locals(func_name="sql.tensor1_sub", positional_args={"t1": y, "t2": s})
        y_ratio = run_on_locals(func_name="sql.tensor1_div", positional_args={"t1": tmp, "t2": d})  # y_ratio = tensor_div(tensor_sub(y, s), d)

        hessian = run_on_locals(func_name="sql.mat_transp_dot_diag_dot_mat", positional_args={"M": X, "d": d}, share_to_global=True)   # hessian = mat_transp_dot_diag_dot_mat(X, d)

        tmp = run_on_locals(func_name="sql.tensor1_add", positional_args={"t1": z, "t2": y_ratio})
        grad = run_on_locals(func_name="sql.mat_transp_dot_diag_dot_vec", positional_args={"M": X, "d": d, "v": tmp}, share_to_global=True)  #grad = mat_transp_dot_diag_dot_vec(X, d, tensor_add(z, y_ratio))

        newlogloss = run_on_locals(func_name="logistic_regression.logistic_loss", positional_args={"v1": y, "v2": s}) # newlogloss = logistic_loss(y, s)
        newlogloss = sum(newlogloss.get_table_data()) # is not a single value, its one value per local node

        # ******** Global part ******** #
        hessian_global = run_on_global(func_name="reduce.sum_tensors",positional_args=[hessian])

        tmp = run_on_global(func_name="logistic_regression.mat_inverse", positional_args=[hessian_global])
        coeff = run_on_global(func_name="sql.matrix_dot_vector", positional_args=[tmp, grad], share_to_locals=True)  # coeff = matrix_dot_vector(mat_inverse(hessian), grad)

        if abs(newlogloss - logloss) <= PREC:
            coeff = run_on_global(func_name="sql.matrix_dot_vector", positional_args=[tmp, grad])  # coeff = matrix_dot_vector(mat_inverse(hessian), grad)
            break
        logloss = newlogloss

    return coeff.get_table_data()


DT = TypeVar("DT")
ND = TypeVar("ND")
S = TypeVar("S")

# SQL UDF
def zeros1(n):
    return numpy.zeros((n,))


# SQL UDF
def matrix_dot_vector(M, v):
    result = M @ v
    return result


@udf
def relation_to_matrix(rel: RelationT[S]) -> TensorT(float, 2):
    return rel


@udf
def relation_to_vector(rel: RelationT[S]) -> TensorT(float, 1):
    return rel


@udf
def tensor_expit(t: TensorT[DT, ND]) -> TensorT[DT, ND]:
    from scipy import special

    result = special.expit(t)
    return result


# SQL UDF
def tensor_mult(t1, t2):
    result = t1 * t2
    return result


# SQL UDF
def tensor_add(t1, t2):
    result = t1 + t2
    return result


# SQL UDF
def tensor_sub(t1, t2):
    result = t1 - t2
    return result


# SQL UDF
def tensor_div(t1, t2):
    result = t1 / t2
    return result


# SQL UDF
def const_tensor_sub(const, t):
    result = const - t
    return result


# SQL UDF
def mat_transp_dot_diag_dot_mat(M, d):
    result = M.T @ numpy.diag(d) @ M
    return result


# SQL UDF
def mat_transp_dot_diag_dot_vec(M, d, v):
    result = M.T @ numpy.diag(d) @ v
    return result


@udf
def logistic_loss(v1: TensorT[DT, ND], v2: TensorT[DT, ND]) -> ScalarT(float):
    from scipy import special

    ll = numpy.sum(special.xlogy(v1, v2) + special.xlogy(1 - v1, 1 - v2))
    return ll


@udf
def tensor_max_abs_diff(t1: TensorT[DT, ND], t2: TensorT[DT, ND]) -> ScalarT(float):
    result = numpy.max(numpy.abs(t1 - t2))
    return result


@udf
def mat_inverse(M: TensorT(Number, 2)) -> TensorT(float, 2):
    minv = numpy.linalg.inv(M)
    return minv


# -------------------------------------------------------- #
# Examples                                                 #
# -------------------------------------------------------- #
def true_run():
    data = pd.read_csv("mipengine/algorithms/auxfiles/logistic_data.csv")
    y = data["alzheimerbroadcategory"].to_numpy()
    X = data[["lefthippocampus", "righthippocampus"]].to_numpy()
    coeff = logistic_regression(y, X)
    print(coeff)


if __name__ == "__main__":
    true_run()
