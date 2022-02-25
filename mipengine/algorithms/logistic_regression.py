# type: ignore
from typing import TypeVar

import numpy
import pandas as pd

from mipengine.algorithm_result_DTOs import TabularDataResult
from mipengine.table_data_DTOs import ColumnDataFloat
from mipengine.table_data_DTOs import ColumnDataStr
from mipengine.udfgen import TensorBinaryOp
from mipengine.udfgen import TensorUnaryOp
from mipengine.udfgen import literal
from mipengine.udfgen import merge_tensor
from mipengine.udfgen import relation
from mipengine.udfgen import tensor
from mipengine.udfgen import udf

PREC = 1e-6


def run(algo_interface):
    local_run = algo_interface.run_udf_on_local_nodes
    global_run = algo_interface.run_udf_on_global_node
    get_table_schema = algo_interface.get_table_schema

    classes = algo_interface.algorithm_parameters["classes"]

    X_relation: "LocalNodeTable" = algo_interface.initial_view_tables["x"]
    y_relation: "LocalNodeTable" = algo_interface.initial_view_tables["y"]

    X = local_run(
        func=relation_to_matrix,
        keyword_args={"rel": X_relation},
    )

    y = local_run(
        func=label_binarize,
        keyword_args={"y": y_relation, "classes": classes},
    )

    # init model
    table_schema = get_table_schema(X)
    ncols = len(table_schema.columns)
    coeff = local_run(
        func=zeros1,
        keyword_args={"n": ncols},
    )

    logloss = 1e6
    while True:
        z = local_run(
            tensor_op=TensorBinaryOp.MATMUL,
            positional_args=[X, coeff],
        )

        s = local_run(
            func=tensor_expit,
            keyword_args={"t": z},
        )

        one_minus_s = local_run(
            tensor_op=TensorBinaryOp.SUB,
            positional_args=[1, s],
        )

        d = local_run(
            tensor_op=TensorBinaryOp.MUL,
            positional_args=[s, one_minus_s],
        )

        y_minus_s = local_run(
            tensor_op=TensorBinaryOp.SUB,
            positional_args=[y, s],
        )

        y_ratio = local_run(
            tensor_op=TensorBinaryOp.DIV,
            positional_args=[y_minus_s, d],
        )

        X_transpose = local_run(
            tensor_op=TensorUnaryOp.TRANSPOSE,
            positional_args=[X],
        )

        d_diag = local_run(
            func=diag,
            keyword_args={"vec": d},
        )

        Xtranspose_dot_ddiag = local_run(
            tensor_op=TensorBinaryOp.MATMUL,
            positional_args=[X_transpose, d_diag],
        )

        hessian = local_run(
            tensor_op=TensorBinaryOp.MATMUL,
            positional_args=[Xtranspose_dot_ddiag, X],
            share_to_global=True,
        )

        z_plus_yratio = local_run(
            tensor_op=TensorBinaryOp.ADD,
            positional_args=[z, y_ratio],
        )

        grad = local_run(
            tensor_op=TensorBinaryOp.MATMUL,
            positional_args=[Xtranspose_dot_ddiag, z_plus_yratio],
            share_to_global=True,
        )

        newlogloss = local_run(
            func=logistic_loss,
            keyword_args={"v1": y, "v2": s},
        )
        # TODO local_run results should not be fetched https://team-1617704806227.atlassian.net/browse/MIP-534
        newlogloss = sum(newlogloss.get_table_data()[1])

        # ~~~~~~~~ Global part ~~~~~~~~ #
        hessian_global = global_run(
            func=sum_tensors,
            keyword_args={"xs": hessian},
        )

        hessian_inverse = global_run(
            func=mat_inverse,
            keyword_args={"m": hessian_global},
        )
        coeff = global_run(
            tensor_op=TensorBinaryOp.MATMUL,
            positional_args=[hessian_inverse, grad],
            share_to_locals=True,
        )

        if abs(newlogloss - logloss) <= PREC:
            coeff = global_run(
                tensor_op=TensorBinaryOp.MATMUL,
                positional_args=[hessian_inverse, grad],
            )
            break
        logloss = newlogloss

    coeff_values = coeff.get_table_data()[2]
    x_variables = algo_interface.x_variables
    result = TabularDataResult(
        title="Logistic Regression Coefficients",
        columns=[
            ColumnDataStr(name="variable", data=x_variables),
            ColumnDataFloat(name="coefficient", data=coeff_values),
        ],
    )
    return result


def logistic_regression_py(y, X, classes):
    y = label_binarize(y, classes)
    X = X.to_numpy()
    nobs, ncols = X.shape
    coeff = zeros1(ncols)
    logloss = 1e6
    while True:
        z = X @ coeff
        s = tensor_expit(z)
        one_minus_s = 1 - s
        d = s * one_minus_s
        y_minus_s = y - s
        y_ratio = y_minus_s / d

        X_transpose = X.T
        d_diag = diag(d)
        Xtranspose_dot_ddiag = X_transpose @ d_diag
        hessian = Xtranspose_dot_ddiag @ X

        z_plus_yratio = z + y_ratio
        grad = Xtranspose_dot_ddiag @ z_plus_yratio
        newlogloss = logistic_loss(y, s)

        hessian_inverse = mat_inverse(hessian)
        coeff = hessian_inverse @ grad

        if abs(newlogloss - logloss) <= PREC:
            break
        logloss = newlogloss
    return coeff


# ~~~~~~~~~~~~~~~~~~~~~~~~ UDFs ~~~~~~~~~~~~~~~~~~~~~~~~~~ #


T = TypeVar("T")
N = TypeVar("N")
S = TypeVar("S")


@udf(y=relation(S), classes=literal(), return_type=tensor(int, 1))
def label_binarize(y, classes):
    from sklearn import preprocessing

    ybin = preprocessing.label_binarize(y, classes=classes).T[0]
    return ybin


@udf(n=literal(), return_type=tensor(float, 1))
def zeros1(n):
    result = numpy.zeros((n,))
    return result


@udf(rel=relation(S), return_type=tensor(float, 2))
def relation_to_matrix(rel):
    return rel


@udf(rel=relation(S), return_type=tensor(float, 1))
def relation_to_vector(rel):
    return rel


@udf(t=tensor(T, N), return_type=tensor(float, N))
def tensor_expit(t):
    from scipy import special

    result = special.expit(t)
    return result


@udf(vec=tensor(T, 1), return_type=tensor(T, 2))
def diag(vec):
    result = numpy.diag(vec)
    return result


@udf(v1=tensor(T, N), v2=tensor(T, N), return_type=relation(schema=[("scalar", float)]))
def logistic_loss(v1, v2):
    from scipy import special

    ll = numpy.sum(special.xlogy(v1, v2) + special.xlogy(1 - v1, 1 - v2))
    return ll


@udf(t1=tensor(T, N), t2=tensor(T, N), return_type=relation(schema=[("scalar", float)]))
def tensor_max_abs_diff(t1, t2):
    result = numpy.max(numpy.abs(t1 - t2))
    return result


@udf(m=tensor(T, 2), return_type=tensor(float, 2))
def mat_inverse(m):
    minv = numpy.linalg.inv(m)
    return minv


@udf(xs=merge_tensor(dtype=T, ndims=N), return_type=tensor(dtype=T, ndims=N))
def sum_tensors(xs):
    from functools import reduce

    reduced = reduce(lambda a, b: a + b, xs)
    reduced = numpy.array(reduced)
    return reduced


def test_logistic_regression():
    data = pd.read_csv("tests/dev_env_tests/data/dementia/edsd.csv")
    y_name = "alzheimerbroadcategory"
    x_names = [
        "lefthippocampus",
        "righthippocampus",
        "rightppplanumpolare",
        "leftamygdala",
        "rightamygdala",
    ]
    classes = ["AD", "CN"]

    data = data[[y_name] + x_names].dropna()
    data = data[data[y_name].isin(classes)]
    y = data[y_name]
    X = data[x_names]

    coeff = logistic_regression_py(y, X, classes=classes)
    expected = numpy.array([4.5790059, -5.680588, -6.193766, 1.807270, 19.584665])
    assert numpy.isclose(coeff, expected).all()
