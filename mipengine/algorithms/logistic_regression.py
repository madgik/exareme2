from typing import DefaultDict
from typing import TypeVar
from typing import Any

import numpy as np
import pandas as pd
from scipy import special

from mipengine.algorithms import udf
from mipengine.algorithms import TableT
from mipengine.algorithms import TensorT
from mipengine.algorithms import LoopbackTableT
from mipengine.algorithms import LiteralParameterT
from mipengine.algorithms import ScalarT
from mipengine.algorithms.patched import expit
from mipengine.algorithms.patched import diag
from mipengine.algorithms.patched import xlogy
from mipengine.algorithms.patched import inv
from mipengine.algorithms.preprocessing import LabelBinarizer
from mipengine.node.udfgen.udfparams import Table
from mipengine.node.udfgen.udfparams import Tensor
from mipengine.node.udfgen.udfparams import LoopbackTable
from mipengine.node.udfgen.udfparams import LiteralParameter
from mipengine.node.udfgen.udfparams import Scalar


PREC = 1e-6


def logistic_regression(y: TableT, X: TableT, classes: LiteralParameterT):
    # init model
    nobs, ncols = X.shape
    coeff = np.zeros(ncols)
    logloss = 1e6
    # binarize labels
    ybin = binarize_labels(y, classes)
    ybin = ybin[:, 0]
    # loop update coefficients
    while True:
        z = matrix_dot_vector(X, coeff)
        s = tensor_expit(z)
        d = tensor_mult(s, const_tensor_sub(1, s))
        y_ratio = tensor_div(tensor_sub(ybin, s), d)

        hessian = mat_transp_dot_diag_dot_mat(X, d)
        grad = mat_transp_dot_diag_dot_vec(X, d, tensor_add(z, y_ratio))
        newlogloss = logistic_loss(ybin, s)
        # ******** Global part ******** #
        coeff = matrix_dot_vector(mat_inverse(hessian), grad)

        if abs(newlogloss - logloss) <= PREC:
            break
        logloss = newlogloss
    return coeff


@udf
def binarize_labels(y: TableT, classes: LiteralParameterT) -> TableT:
    binarizer = LabelBinarizer()
    binarizer.fit(classes)
    binarized = binarizer.transform(y)
    return binarized


@udf
def matrix_dot_vector(M: TensorT, v: LiteralParameterT) -> TensorT:
    result = M @ v
    return result


@udf
def tensor_expit(t: TensorT) -> TensorT:
    result = expit(t)
    return result


@udf
def tensor_mult(t1: TensorT, t2: TensorT) -> TensorT:
    result = t1 * t2
    return result


@udf
def tensor_add(t1: TensorT, t2: TensorT) -> TensorT:
    result = t1 + t2
    return result


@udf
def tensor_sub(t1: TensorT, t2: TensorT) -> TensorT:
    result = t1 - t2
    return result


@udf
def tensor_div(t1: TensorT, t2: TensorT) -> TensorT:
    result = t1 / t2
    return result


@udf
def const_tensor_sub(const: LiteralParameterT, t: TensorT) -> TensorT:
    result = const - t
    return result


@udf
def mat_transp_dot_diag_dot_mat(M: TensorT, d: TensorT) -> TensorT:
    result = M.T @ diag(d) @ M
    return result


@udf
def mat_transp_dot_diag_dot_vec(M: TensorT, d: TensorT, v: TensorT) -> TensorT:
    result = M.T @ diag(d) @ v
    return result


@udf
def logistic_loss(v1: TensorT, v2: TensorT) -> ScalarT:
    ll = np.sum(xlogy(v1, v2) + xlogy(1 - v1, 1 - v2))
    return ll


@udf
def tensor_max_abs_diff(t1: TensorT, t2: TensorT) -> ScalarT:
    result = np.max(np.abs(t1 - t2))
    return result


@udf
def mat_inverse(M: TensorT) -> TensorT:
    minv = inv(M)
    return minv


# -------------------------------------------------------- #
# Examples                                                 #
# -------------------------------------------------------- #
def true_run():
    data = pd.read_csv("mipengine/algorithms/logistic_data.csv")
    y = data["alzheimerbroadcategory"].to_numpy()
    X = data[["lefthippocampus", "righthippocampus"]].to_numpy()
    coeff = logistic_regression(y, X, np.array(["AD", "CN"]))
    print(coeff)


# def mock_run():
#     y = Table(dtype=str, shape=(1000, 1))
#     X = Table(dtype=float, shape=(1000, 2))
#     classes = LiteralParameter(np.array(["AD", "CN"]))
#     coeff = logistic_regression(y, X, classes)
#     print(coeff)


if __name__ == "__main__":
    true_run()
    # mock_run()
