from mipengine.algorithms.patched import diag
from mipengine.algorithms.patched import expit
from mipengine.algorithms.patched import xlogy
from mipengine.algorithms.patched import inv
from mipengine.node.udfgen.udfparams import DatalessArray


def true_dimensions(table):
    if 1 in table.shape:
        return len([_ for _ in table.shape if _ != 1])
    return len(table.shape)


@expit.register
def _(x: DatalessArray):
    return DatalessArray(float, x.shape)


@diag.register
def _(v: DatalessArray, k=0):
    if true_dimensions(v) == 1:
        return DatalessArray(v.dtype, shape=(v.shape[0], v.shape[0]))
    elif true_dimensions(v) == 2:
        if v.shape[0] == v.shape[1]:
            diag_length = min(v.shape) - k
            return DatalessArray(v.dtype, shape=(diag_length, 1))
        else:
            raise NotImplementedError


@xlogy.register
def _(x: DatalessArray, y: DatalessArray):
    outshape = (x + y).shape
    return DatalessArray(float, outshape)


@inv.register
def _(x: DatalessArray):
    return DatalessArray(float, x.shape)
