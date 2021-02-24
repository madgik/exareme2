from typing import DefaultDict

import numpy as np
import pandas as pd

from mipengine.algorithms.numpy2 import diag
from mipengine.algorithms.numpy2 import zeros
from mipengine.algorithms.numpy2 import inv
from mipengine.algorithms.preprocessing import LabelBinarizer
from mipengine.algorithms.specialfuncs import expit
from mipengine.algorithms.specialfuncs import xlogy
from mipengine.algorithms.udfgen.udfgenerator import monet_udf
from mipengine.algorithms.udfgen.udfparams import Table
from mipengine.algorithms.udfgen.udfparams import Tensor
from mipengine.algorithms.udfgen.udfparams import LoopbackTable
from mipengine.algorithms.udfgen.udfparams import LiteralParameter
from mipengine.algorithms.udfgen.udfparams import Scalar

PREC = 1e-6


def logistic_regression_true(y: Table, X: Table, classes: LiteralParameter):
    # init model
    nobs, ncols = X.shape
    coeff = init_tensor_zeros((ncols,))
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

        invhessian = mat_inverse(hessian)
        coeff = matrix_dot_vector(invhessian, grad)

        if abs(newlogloss - logloss) <= PREC:
            break
        logloss = newlogloss
    return coeff


def logistic_regression_mock(y: Table, X: Table, classes: LiteralParameter):
    # init model
    nobs, ncols = X.shape
    coeff = init_tensor_zeros(LiteralParameter((ncols,)))
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

        invhessian = mat_inverse(hessian)
        coeff = matrix_dot_vector(invhessian, grad)

        if abs(newlogloss - logloss) <= PREC:
            break
        logloss = newlogloss
    return coeff


def logistic_regression_genudf(y: Table, X: Table, classes: LiteralParameter):
    udefs = []
    # init model
    nobs, ncols = X.shape
    coeff = init_tensor_zeros(LiteralParameter((ncols,)))
    udefs.append(
        init_tensor_zeros.to_sql("init_tensor_zeros", LiteralParameter((ncols, 1)))
    )
    logloss = 1e6
    # binarize labels
    ybin = binarize_labels(y, classes)
    udefs.append(binarize_labels.to_sql("binarize_labels", y, classes))
    ybin = ybin[:, 0]
    # loop update coefficients
    while True:
        z = matrix_dot_vector(X, coeff)
        udefs.append(
            matrix_dot_vector.to_sql(
                "matrix_dot_vector",
                X,
                LoopbackTable(name="coeff", dtype=coeff.dtype, shape=coeff.shape),
            )
        )
        s = tensor_expit(z)
        udefs.append(tensor_expit.to_sql("tensor_expit", z))
        t1 = const_tensor_sub(1, s)
        udefs.append(
            const_tensor_sub.to_sql("const_tensor_sub", LiteralParameter(1), s)
        )
        d = tensor_mult(s, t1)
        udefs.append(tensor_mult.to_sql("tensor_mult", s, t1))
        t2 = tensor_sub(ybin, s)
        udefs.append(tensor_sub.to_sql("tensor_sub", ybin, s))
        y_ratio = tensor_div(t2, d)
        udefs.append(tensor_div.to_sql("tensor_div", t2, d))

        hessian = mat_transp_dot_diag_dot_mat(X, d)
        udefs.append(
            mat_transp_dot_diag_dot_mat.to_sql("mat_transp_dot_diag_dot_mat", X, d)
        )
        t3 = tensor_add(z, y_ratio)
        udefs.append(tensor_add.to_sql("tensor_add", z, y_ratio))
        grad = mat_transp_dot_diag_dot_vec(X, d, t3)
        udefs.append(
            mat_transp_dot_diag_dot_vec.to_sql("mat_transp_dot_diag_dot_vec", X, d, t3)
        )
        newlogloss = logistic_loss(ybin, s)
        udefs.append(logistic_loss.to_sql("logistic_loss", ybin, s))

        invhessian = mat_inverse(hessian)
        udefs.append(mat_inverse.to_sql("mat_inverse", hessian))
        coeff = matrix_dot_vector(invhessian, grad)
        udefs.append(
            matrix_dot_vector.to_sql(
                "matrix_dot_vector",
                invhessian,
                LoopbackTable("grad", grad.dtype, grad.shape),
            )
        )

        if abs(newlogloss - logloss) <= PREC:
            break
        logloss = newlogloss
    return udefs


@monet_udf
def init_tensor_zeros(shape: LiteralParameter) -> Tensor:
    z = zeros(shape)
    return z


@monet_udf
def binarize_labels(y: Table, classes: LiteralParameter) -> Table:
    binarizer = LabelBinarizer()
    binarizer.fit(classes)
    binarized = binarizer.transform(y)
    return binarized


@monet_udf
def matrix_dot_vector(M: Tensor, v: LoopbackTable) -> Tensor:
    result = M @ v
    return result


@monet_udf
def tensor_expit(t: Tensor) -> Tensor:
    result = expit(t)
    return result


@monet_udf
def tensor_mult(t1: Tensor, t2: Tensor) -> Tensor:
    result = t1 * t2
    return result


@monet_udf
def tensor_add(t1: Tensor, t2: Tensor) -> Tensor:
    result = t1 + t2
    return result


@monet_udf
def tensor_sub(t1: Tensor, t2: Tensor) -> Tensor:
    result = t1 - t2
    return result


@monet_udf
def tensor_div(t1: Tensor, t2: Tensor) -> Tensor:
    result = t1 / t2
    return result


@monet_udf
def const_tensor_sub(const: LiteralParameter, t: Tensor) -> Tensor:
    result = const - t
    return result


@monet_udf
def mat_transp_dot_diag_dot_mat(M: Tensor, d: Tensor) -> Tensor:
    result = M.T @ diag(d) @ M
    return result


@monet_udf
def mat_transp_dot_diag_dot_vec(M: Tensor, d: Tensor, v: Tensor) -> Tensor:
    result = M.T @ diag(d) @ v
    return result


@monet_udf
def logistic_loss(v1: Tensor, v2: Tensor) -> Scalar:
    ll = np.sum(xlogy(v1, v2) + xlogy(1 - v1, 1 - v2))
    return ll


@monet_udf
def tensor_max_abs_diff(t1: Tensor, t2: Tensor) -> Scalar:
    result = np.max(np.abs(t1 - t2))
    return result


@monet_udf
def mat_inverse(M: Tensor) -> Tensor:
    minv = inv(M)
    return minv


# -------------------------------------------------------- #
# Examples                                                 #
# -------------------------------------------------------- #
def true_run():
    data = pd.read_csv("mipengine/algorithms/logistic_data.csv")
    y = data["alzheimerbroadcategory"].to_numpy()
    X = data[["lefthippocampus", "righthippocampus"]].to_numpy()
    coeff = logistic_regression_true(y, X, np.array(["AD", "CN"]))
    print(coeff)


def mock_run():
    y = Table(dtype=str, shape=(1000, 1))
    X = Table(dtype=float, shape=(1000, 2))
    classes = LiteralParameter(np.array(["AD", "CN"]))
    coeff = logistic_regression_mock(y, X, classes)
    print(coeff)


def genudf_run():
    y = Table(dtype=str, shape=(1000, 1))
    X = Table(dtype=float, shape=(1000, 2))
    classes = LiteralParameter(np.array(["AD", "CN"]))
    udefs = logistic_regression_genudf(y, X, classes)
    for udef in udefs:
        print(udef)
        print()


if __name__ == "__main__":
    true_run()
    mock_run()
    genudf_run()
