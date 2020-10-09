import re
import inspect
from typing import Tuple
from typing import get_args
from typing import get_origin
from typing import get_type_hints


import numpy as np

FUNC_DEFS = []
PY_TO_SQL_TYPES = {int: "INT", float: "REAL"}


def py_to_sql_type(annotation):
    if annotation == int:
        return PY_TO_SQL_TYPES[annotation]
    elif annotation == float:
        return PY_TO_SQL_TYPES[annotation]
    elif get_origin(annotation) == tuple:
        params = get_args(annotation)
        params = ", ".join(
            [str(i) + " " + PY_TO_SQL_TYPES[param] for i, param in enumerate(params)]
        )
        return f"TABLE({params})"


class MetaAlgorithm(type):
    """Metaclass for Algorithm base class. Every time a subclass of Algorithm
    is defined, all its methods become MonetDB UDFs.
    """

    def __new__(mcs, name, bases, attrs):
        for key, attr in attrs.items():
            if callable(attr):
                register(attr)
        return type.__new__(mcs, name, bases, attrs)


def register(func):
    """Registers a function as MonetDB UDF."""
    types = get_type_hints(func)
    name = func.__qualname__
    params = [
        name + " " + py_to_sql_type(type_)
        for name, type_ in types.items()
        if name != "return"
    ]
    params = ", ".join(params)
    return_type = py_to_sql_type(types["return"])
    sourcelines = inspect.getsourcelines(func)[0]
    body = get_body(sourcelines)
    func_def = f"CREATE OR REPLACE FUNCTION {name}({params})\n"
    func_def += f"RETURNS {return_type}\n"
    func_def += "LANGUAGE PYTHON {\n"
    func_def += body
    func_def += "};"
    FUNC_DEFS.append(func_def)


def get_body(sourcelines):
    """
    :sourcelines: result of inspect.getsourcelines on a function
    :returns: body of function as a single long string
    """
    iterlines = iter(sourcelines)
    next(line for line in iterlines if "->" in line)  # hack to remove func definition
    lines = (re.sub(r"^\s{4}", "", line) for line in iterlines)  # remove indents
    return "".join(lines)


class Algorithm(metaclass=MetaAlgorithm):
    """Abstract base class, to be subclassed by all algorithms."""

    pass


# ------------------------------------------------------------
# Algorithm example
# ------------------------------------------------------------
class Pearson(Algorithm):
    def local(x: float, y: float) -> Tuple[float, float, float, float, float, int]:
        X = x
        Y = y
        sx = X.sum(axis=0)
        sy = Y.sum(axis=0)
        sxx = (X ** 2).sum(axis=0)
        syy = (Y ** 2).sum(axis=0)
        sxy = (X * Y).sum(axis=0)
        n = X.size
        return sx, sy, sxx, syy, sxy, n

    def global_(
        sx: float, sxx: float, sxy: float, sy: float, syy: float, n: int
    ) -> float:
        n = np.sum(n)
        sx = np.sum(sx)
        sxx = np.sum(sxx)
        sxy = np.sum(sxy)
        sy = np.sum(sy)
        syy = np.sum(syy)
        d = np.sqrt(n * sxx - sx * sx) * np.sqrt(n * syy - sy * sy)
        return (n * sxy - sx * sy) / d
