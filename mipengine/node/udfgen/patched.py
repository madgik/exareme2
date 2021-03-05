from mipengine.algorithms.patched import diag
from mipengine.algorithms.patched import expit
from mipengine.algorithms.patched import xlogy
from mipengine.algorithms.patched import inv
from mipengine.node.udfgen.udfparams import Table


def true_dimensions(table):
    if 1 in table.shape:
        return len([_ for _ in table.shape if _ != 1])
    return len(table.shape)


@expit.register
def _(x: Table):
    return Table(float, x.shape)


@diag.register
def _(v: Table, k=0):
    if true_dimensions(v) == 1:
        return Table(v.dtype, shape=(v.shape[0], v.shape[0]))
    elif true_dimensions(v) == 2:
        if v.shape[0] == v.shape[1]:
            diag_length = min(v.shape) - k
            return Table(v.dtype, shape=(diag_length, 1))
        else:
            raise NotImplementedError


@xlogy.register
def _(x: Table, y: Table):
    outshape = (x + y).shape
    return Table(float, outshape)


@inv.register
def _(x: Table):
    return Table(float, x.shape)
