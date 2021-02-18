from typing import DefaultDict

import numpy as np

from mipengine.algorithms.numpy2 import diag
from mipengine.algorithms.numpy2 import zeros
from mipengine.algorithms.preprocessing import LabelBinarizer
from mipengine.algorithms.specialfuncs import expit
from mipengine.algorithms.specialfuncs import xlogy
from mipengine.node import monet_udf
from mipengine.node import Table
from mipengine.node import Tensor
from mipengine.node import LoopbackTable
from mipengine.node import LiteralParameter
from mipengine.node import Scalar

PREC = 1e-6


def logistic_regression(y: Table, X: Table, classes: LiteralParameter):
    # init model
    nobs, ncols = X.shape
    coeff = init_tensor_zeros(ncols)
    logloss = 1e6
    # binarize labels
    ybin = binarize_labels(y, classes)
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
    minv = np.linalg.inv(M)
    return minv


# -------------------------------------------------------- #
# Examples                                                 #
# -------------------------------------------------------- #
