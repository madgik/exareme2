def udf(func):
    func.registered = True
    func.scalar = True
    func.aggregate = False
    func.table = False
    func.jit = False
    return func


def udaf(func):
    func.registered = True
    func.aggregate = True
    func.scalar = False
    func.table = False
    func.jit = False
    return func


def udtf(func):
    func.registered = True
    func.table = True
    func.aggregate = False
    func.scalar = False
    func.jit = False
    return func


def jitudf(func):
    func.registered = True
    func.scalar = True
    func.aggregate = False
    func.table = False
    func.jit = True
    return func


def jitudaf(func):
    func.registered = True
    func.aggregate = True
    func.scalar = False
    func.table = False
    func.jit = True
    return func


def jitudtf(func):
    func.registered = True
    func.table = True
    func.aggregate = False
    func.scalar = False
    func.jit = True
    return func
