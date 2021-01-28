# from types import ModuleType

import numpy


def get_ufuncs(module):
    ufuncs = {}
    for name, obj in module.__dict__.items():
        if type(obj) == numpy.ufunc:
            ufuncs[name] = obj
    return ufuncs


ufuncs = get_ufuncs(numpy)


# def get_types(module):
#     types = ()
#     for name in dir(module):
#         if type(getattr(module, name)) == type:
#             types += (getattr(module, name),)
#         elif type(getattr(module, name)) == ModuleType:
#             types += get_types(getattr(module, name))
#     return types


# types = get_types(numpy)

types = (
    bool,
    int,
    float,
    complex,
    numpy.bool_,
    numpy.bool8,
    numpy.byte,
    numpy.short,
    numpy.intc,
    numpy.int_,
    numpy.longlong,
    numpy.intp,
    numpy.int8,
    numpy.int16,
    numpy.int32,
    numpy.int64,
    numpy.ubyte,
    numpy.ushort,
    numpy.uintc,
    numpy.uint,
    numpy.ulonglong,
    numpy.uintp,
    numpy.uint8,
    numpy.uint16,
    numpy.uint32,
    numpy.uint64,
    numpy.half,
    numpy.single,
    numpy.double,
    numpy.float_,
    numpy.longfloat,
    numpy.float16,
    numpy.float32,
    numpy.float64,
    numpy.float128,
    numpy.csingle,
    numpy.complex_,
    numpy.clongfloat,
    numpy.complex64,
    numpy.complex128,
    numpy.complex256,
)


def compute_type_conversion_table(ufunc):
    type_conversion_table = {}
    if ufunc.__name__ == "arctanh":
        val = 0
    else:
        val = 1
    if ufunc.nin == 1:
        for tp in types:
            try:
                outtype = type(ufunc(tp(val)))
            except TypeError:
                continue
            else:
                type_conversion_table[(tp,)] = outtype
    elif ufunc.nin == 2:
        for tp_1 in types:
            for tp_2 in types:
                try:
                    outtype = type(ufunc(tp_1(1), tp_2(1)))
                except TypeError:
                    continue
                else:
                    type_conversion_table[(tp_1, tp_2)] = outtype
    return type_conversion_table


type_conversion_table = {}
for name, ufunc in ufuncs.items():
    if name == "matmul":
        continue
    conversions = compute_type_conversion_table(ufunc)
    type_conversion_table[name] = conversions

type_conversion_table["matmul"] = {}
for tp_1 in types:
    for tp_2 in types:
        try:
            matrix_1 = numpy.array([[1, 2, 3], [1, 2, 3]], dtype=tp_1).T
            matrix_2 = numpy.array([[1, 2, 3], [1, 2, 3]], dtype=tp_2)
        except TypeError:
            continue
        else:
            out = matrix_1 @ matrix_2
            type_conversion_table["matmul"][(tp_1, tp_2)] = type(out[0, 0])
