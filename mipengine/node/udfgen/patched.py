from mipengine.algorithms.patched import diag
from mipengine.algorithms.patched import expit
from mipengine.algorithms.patched import xlogy
from mipengine.algorithms.patched import inv
from mipengine.node.udfgen.datalesstypes import DatalessTensor


def true_dimensions(table):
    if 1 in table.shape:
        return len([_ for _ in table.shape if _ != 1])
    return len(table.shape)


@expit.register
def _(x: DatalessTensor):
    return DatalessTensor(float, x.shape)


@diag.register
def _(v: DatalessTensor, k=0):
    if true_dimensions(v) == 1:
        return DatalessTensor(v.dtype, shape=(v.shape[0], v.shape[0]))
    elif true_dimensions(v) == 2:
        if v.shape[0] == v.shape[1]:
            diag_length = min(v.shape) - k
            return DatalessTensor(v.dtype, shape=(diag_length, 1))
        else:
            raise NotImplementedError


@xlogy.register
def _(x: DatalessTensor, y: DatalessTensor):
    outshape = (x + y).shape
    return DatalessTensor(float, outshape)


@inv.register
def _(x: DatalessTensor):
    return DatalessTensor(float, x.shape)
