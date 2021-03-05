from functools import singledispatch
from functools import wraps

import numpy as np
import pandas as pd
from scipy import special


def patch(func, *types):
    @wraps(func)
    @singledispatch
    def patched(*args):
        msg = f"Function {func.__name__} has no implementation for type {type(args[0])}"
        raise NotImplementedError(msg)

    def dispatched(*args):
        return func(*args)

    for tp in types:
        patched.register(tp, dispatched)

    return patched


xlogy = patch(special.xlogy, np.ndarray)
diag = patch(np.diag, np.ndarray)
expit = patch(special.expit, np.ndarray, pd.DataFrame)
inv = patch(np.linalg.inv, np.ndarray)
