import re
import inspect
from functools import wraps
from typing import get_type_hints
from typing import get_origin
from typing import get_args

import numpy as np

FUNC_DEFS = []
PY_TO_SQL_TYPES = {
    np.dtype(np.int32): "INT",
    np.dtype(np.int64): "INT",
    np.dtype(np.float32): "REAL",
    np.dtype(np.float64): "REAL",
    int: "INT",
    float: "REAL",
}


class MetaAlgorithm(type):
    """Metaclass for Algorithm base class. Every time a subclass of Algorithm
    is defined, all its methods are decorated with make_udf. Then, they can
    emit a UDF code on call time.
    """

    def __new__(mcs, name, bases, attrs):
        attrs = {key: make_udf(attr) for key, attr in attrs.items() if callable(attr)}
        return type.__new__(mcs, name, bases, attrs)


class Algorithm(metaclass=MetaAlgorithm):
    """Abstract base class, to be subclassed by all algorithms."""

    pass


def make_udf(func):
    verify_annotations(func)  # verify at compile time

    @wraps(func)
    def wrapper(*args):  # no kwargs for simplicity
        udf_params = get_udf_params(func, args)
        register(func, udf_params)
        return func(*args)

    return wrapper


def verify_annotations(func):
    """Verifies that func is well annotated. All algorithm methods should be
    fully annotated (parameters and return val) in order to become UDFs.
    """
    parameters = inspect.signature(func).parameters
    annotations = func.__annotations__
    parameters = dict(parameters)
    del parameters["self"]
    if len(parameters) != len(annotations) - 1:  # subtract return annotation
        msg = "This method should be fully annotated. "
        msg += f"Some annotations are missing: {annotations}"
        raise SyntaxError(msg)


def get_udf_params(func, args):
    """Extracts UDF parameters from python parameters dynamically.  Since
    python methods accept multidimensional arrays whereas MonetDB UDFs only work
    with one-dimensional ones, we need to create the UDF at call time.

    Args:
      func:
      args:

    Returns:
        This method counts the number of columns in python parameters and
        outputs a list of formatted SQL parameters of appropriate length

    """
    signature = inspect.signature(func)
    param_names = signature.parameters.keys()
    sql_params = []
    for name, arg in zip(param_names, args):
        if name == "self":
            continue
        sql_type = PY_TO_SQL_TYPES[arg.dtype]
        if len(arg.shape) == 1:
            num_cols = 1
        elif len(arg.shape) == 2:
            num_cols = arg.shape[1]
        sql_params += [name + str(i) + " " + sql_type for i in range(num_cols)]
    return sql_params


def register(func, params):
    """Registers a function as MonetDB UDF."""
    global FUNC_DEFS
    types = get_type_hints(func)
    name = func.__qualname__
    params = ", ".join(params)
    return_type = get_udf_return_type(types["return"])
    sourcelines = inspect.getsourcelines(func)[0]
    body = get_body(sourcelines)
    func_def = f"CREATE OR REPLACE FUNCTION {name}({params})\n"
    func_def += f"RETURNS {return_type}\n"
    func_def += "LANGUAGE PYTHON {\n"
    func_def += body
    func_def += "};"
    FUNC_DEFS.append(func_def)


def get_udf_return_type(annotation):
    if get_origin(annotation) == tuple:
        params = get_args(annotation)
        params = [
            "ret" + str(i) + " " + PY_TO_SQL_TYPES[param]
            for i, param in enumerate(params)
        ]
        params = ", ".join(params)
        return f"TABLE({params})"
    else:
        return PY_TO_SQL_TYPES[annotation]


def get_body(sourcelines):
    """

    Args:
      sourcelines: result of inspect.getsourcelines on a function

    Returns:
      body of function as a single long string

    """
    iterlines = iter(sourcelines)
    next(line for line in iterlines if "->" in line)  # hack to remove func definition
    lines = (re.sub(r"^\s{4}", "", line) for line in iterlines)  # remove indents
    return "".join(lines)


# ------------------------------------------------------------
# Algorithm example
# ------------------------------------------------------------
class TheAlgorithm(Algorithm):
    def the_method(self, x: np.ndarray, y: np.ndarray) -> int:
        return 1


if __name__ == "__main__":
    alg = TheAlgorithm()

    X = np.array([[1, 2, 3, 4], [10, 20, 30, 40]])
    Y = np.array([[5, 6, 7, 8], [50, 60, 70, 80]])
    print(FUNC_DEFS)  # should be empty: FUNC_DEFS is populated at call time
    alg.the_method(X, Y)
    print(FUNC_DEFS[0])

    X = np.array([[1.5, 2], [10, 20]])
    Y = np.array([[5, 6, 7], [50, 60, 70]])
    alg.the_method(X, Y)
    print(FUNC_DEFS[1])
