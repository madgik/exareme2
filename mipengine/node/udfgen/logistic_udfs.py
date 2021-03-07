import numpy as np

from mipengine.algorithms.logistic_regression import binarize_labels
from mipengine.algorithms.logistic_regression import matrix_dot_vector
from mipengine.algorithms.logistic_regression import tensor_mult
from mipengine.algorithms.logistic_regression import const_tensor_sub
from mipengine.algorithms.logistic_regression import tensor_expit
from mipengine.algorithms.logistic_regression import tensor_div
from mipengine.algorithms.logistic_regression import tensor_sub
from mipengine.algorithms.logistic_regression import mat_transp_dot_diag_dot_mat
from mipengine.algorithms.logistic_regression import mat_transp_dot_diag_dot_vec
from mipengine.algorithms.logistic_regression import tensor_add
from mipengine.algorithms.logistic_regression import logistic_loss
from mipengine.algorithms.logistic_regression import mat_inverse
import mipengine.node.udfgen.patched
from mipengine.node.udfgen.udfgenerator import get_generator
from mipengine.node.udfgen.udfgenerator import generate_udf
from mipengine.node.udfgen.udfparams import DatalessTensor
from mipengine.node.udfgen.udfparams import LoopbackTable
from mipengine.node.udfgen.udfparams import LiteralParameter


def logistic_regression_udfs(y: DatalessTensor, X: DatalessTensor, classes: LiteralParameter):
    udfs = []
    # init model
    nobs, ncols = X.shape
    coeff = np.zeros(ncols)
    logloss = 1e6
    # binarize labels
    ybin = binarize_labels(y, classes)
    udf = generate_udf(
        "logistic_regression.binarize_labels",
        "logistic_regression_binarize_labels_1234",
        to_arg_descr(y, classes),
        {},
    )
    udfs.append(udf)

    ybin = ybin[:, 0]
    # loop update coefficients
    z = matrix_dot_vector(X, coeff)
    udf = generate_udf(
        "logistic_regression.matrix_dot_vector",
        "logistic_regression_matrix_dot_vector_1234",
        to_arg_descr(X, coeff),
        {},
    )
    udfs.append(udf)

    s = tensor_expit(z)
    udf = generate_udf(
        "logistic_regression.tensor_expit",
        "logistic_regression_tensor_expit_1234",
        to_arg_descr(z),
        {},
    )
    udfs.append(udf)

    t1 = const_tensor_sub(1, s)
    udf = generate_udf(
        "logistic_regression.const_tensor_sub",
        "logistic_regression_const_tensor_sub_1234",
        to_arg_descr(1, s),
        {},
    )
    udfs.append(udf)

    d = tensor_mult(s, t1)
    udf = generate_udf(
        "logistic_regression.tensor_mult",
        "logistic_regression_tensor_mult_1234",
        to_arg_descr(s, t1),
        {},
    )
    udfs.append(udf)

    t2 = tensor_sub(ybin, s)
    udf = generate_udf(
        "logistic_regression.tensor_sub",
        "logistic_regression_tensor_sub_1234",
        to_arg_descr(ybin, s),
        {},
    )
    udfs.append(udf)

    y_ratio = tensor_div(t2, d)
    udf = generate_udf(
        "logistic_regression.tensor_div",
        "logistic_regression_tensor_div_1234",
        to_arg_descr(t2, d),
        {},
    )
    udfs.append(udf)

    hessian = mat_transp_dot_diag_dot_mat(X, d)
    udf = generate_udf(
        "logistic_regression.mat_transp_dot_diag_dot_mat",
        "logistic_regression_mat_transp_dot_diag_dot_mat_1234",
        to_arg_descr(X, d),
        {},
    )
    udfs.append(udf)

    t3 = tensor_add(z, y_ratio)
    udf = generate_udf(
        "logistic_regression.tensor_add",
        "logistic_regression_tensor_add_1234",
        to_arg_descr(z, y_ratio),
        {},
    )
    udfs.append(udf)

    grad = mat_transp_dot_diag_dot_vec(X, d, t3)
    udf = generate_udf(
        "logistic_regression.mat_transp_dot_diag_dot_vec",
        "logistic_regression_mat_transp_dot_diag_dot_vec_1234",
        to_arg_descr(X, d, t3),
        {},
    )
    udfs.append(udf)

    newlogloss = logistic_loss(ybin, s)
    udf = generate_udf(
        "logistic_regression.logistic_loss",
        "logistic_regression_logistic_loss_1234",
        to_arg_descr(ybin, s),
        {},
    )
    udfs.append(udf)

    # ******** Global part ******** #
    t4 = mat_inverse(hessian)
    udf = generate_udf(
        "logistic_regression.mat_inverse",
        "logistic_regression_mat_inverse_1234",
        to_arg_descr(hessian),
        {},
    )
    udfs.append(udf)

    coeff = matrix_dot_vector(
        t4, grad
    )  # TODO to make this work we need to coerce Table to LiteralParameter
    # udf = generate_udf('logistic_regression.matrix_dot_vector', 'logistic_regression_matrix_dot_vector_1234', to_arg_descr(t4, grad), {})
    # udfs.append(udf)

    return udfs


def to_table_descr(table):
    return {
        "type": "input_table",
        "schema": [{"type": table.dtype.__name__} for _ in range(table.ncols)],
        "nrows": table.shape[0],
    }


def array_to_table_descr(array):
    return to_table_descr(LoopbackTable("coeff", float, array.shape))


def to_loopback_descr(table):
    return {
        "type": "loopback_table",
        "name": table.name,
        "schema": [{"type": table.dtype.__name__} for _ in range(table.ncols)],
        "nrows": table.shape[0],
    }


def to_literal_param(literal):
    return {"type": "literal_parameter", "value": literal.value}


def to_arg_descr(*args):
    descriptions = []
    for arg in args:
        if type(arg) == DatalessTensor:
            descriptions.append(to_table_descr(arg))
        elif type(arg) == LoopbackTable:
            descriptions.append(to_loopback_descr(arg))
        elif type(arg) == LiteralParameter:
            descriptions.append(to_literal_param(arg))
        elif type(arg) == np.ndarray:  # TODO print literal np.ndarray in UDFGenerator
            descriptions.append(to_literal_param(LiteralParameter(arg)))
        else:
            descriptions.append(to_literal_param(LiteralParameter(arg)))
    return descriptions


def mock_run():
    y = DatalessTensor(dtype=str, shape=(1000, 1))
    X = DatalessTensor(dtype=float, shape=(1000, 2))
    classes = LiteralParameter(np.array(["AD", "CN"]))
    udfs = logistic_regression_udfs(y, X, classes)
    for udf in udfs:
        print(udf)
        print("-" * 80)


if __name__ == "__main__":
    mock_run()
