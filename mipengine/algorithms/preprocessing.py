from functools import singledispatchmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn import preprocessing

from mipengine.node.udfgen.udfparams import Table
from mipengine.node.udfgen.udfparams import LiteralParameter


class LabelBinarizer(preprocessing.LabelBinarizer):
    @singledispatchmethod
    def fit(self, y):
        raise NotImplementedError

    @fit.register
    def _(self, y: np.ndarray):
        return super().fit(y)

    @fit.register
    def _(self, y: LiteralParameter):
        return super().fit(y.value)

    @singledispatchmethod
    def transform(self, y):
        raise NotImplementedError

    @transform.register
    def _(self, y: np.ndarray):
        return super().transform(y)

    @transform.register
    def _(self, y: Table):
        if len(self.classes_) == 2:
            return Table(dtype=int, shape=(y.shape[0], 1))
        else:
            return Table(dtype=int, shape=(y.shape[0], len(self.classes_)))
