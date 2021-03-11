# type: ignore
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

PREC = 1e-6


def logistic_regression(y: TableT, X: TableT, classes: LiteralParameterT):
    # init model
    nobs, ncols = X.shape
    coeff = zeros((ncols,))
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


DT = TypeVar("DT")
ND = TypeVar("ND")

# TODO SQL UDF
def zeros(shape):
    return numpy.zeros(shape)


@udf
def binarize_labels(y: TableT, classes: LiteralParameterT) -> TensorT(int, 1):
    from sklearn.preprocessing import LabelBinarizer

    binarizer = LabelBinarizer()
    binarizer.fit(classes)
    binarized = binarizer.transform(y)
    return binarized


# TODO SQL UDF
def matrix_dot_vector(M, v):
    result = M @ v
    return result


@udf
def tensor_expit(t: TensorT[DT, ND]) -> TensorT[DT, ND]:
    from scipy import special

    result = special.expit(t)
    return result


# TODO SQL UDF
def tensor_mult(t1, t2):
    result = t1 * t2
    return result


# TODO SQL UDF
def tensor_add(t1, t2):
    result = t1 + t2
    return result


# TODO SQL UDF
def tensor_sub(t1, t2):
    result = t1 - t2
    return result


# TODO SQL UDF
def tensor_div(t1, t2):
    result = t1 / t2
    return result


# TODO SQL UDF
def const_tensor_sub(const, t):
    result = const - t
    return result


# TODO SQL UDF
def mat_transp_dot_diag_dot_mat(M, d):
    result = M.T @ numpy.diag(d) @ M
    return result


# TODO SQL UDF
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
def mat_inverse(M: TensorT[DT, ND]) -> TensorT[DT, ND]:
    minv = numpy.linalg.inv(M)
    return minv


# -------------------------------------------------------- #
# Examples                                                 #
# -------------------------------------------------------- #
def true_run():
    data = pd.read_csv("mipengine/algorithms/logistic_data.csv")
    y = data["alzheimerbroadcategory"].to_numpy()
    X = data[["lefthippocampus", "righthippocampus"]].to_numpy()
    coeff = logistic_regression(y, X, numpy.array(["AD", "CN"]))
    print(coeff)


if __name__ == "__main__":
    true_run()
