from numbers import Number
import inspect
import ast
import random

import numpy as np


# -------------------------------------------------------- #
# UDF                                                      #
# -------------------------------------------------------- #
SQLTYPES = {
    int: "BIGINT",
    float: "DOUBLE",
    str: "TEXT",
    np.dtype('int32'): "INT",
    np.dtype('int64'): "BIGINT",
    np.dtype('float32'): "FLOAT",
    np.dtype('float64'): "DOUBLE",
}


class UDFFactory:
    func_bodies = {}
    func_decls = {}
    udf_table = {}

    def __init__(self, func):
        verify_annotations(func)
        self.func = func
        self.name = func.__name__
        self.annotations = func.__annotations__
        self.output_type = self.annotations['return']
        self._register_func(func)

    @classmethod
    def _register_func(cls, func):
        name = func.__name__
        declaration, body = parse_func(func)
        cls.func_decls[name] = declaration
        cls.func_bodies[name] = body

    def _get_output_type(self):
        if isinstance(self.output_type, _Table):
            return self.output_type.types[0]
        elif issubclass(self.output_type, Number):
            return self.output_type
        else:
            raise ValueError('Wat??')

    def _get_return_name(self):
        code = self.func_decls[self.name] + "\n".join(self.func_bodies[self.name])
        tree = ast.parse(code)
        body_stmts = tree.body[0].body
        ret_idx = [isinstance(s, ast.Return) for s in body_stmts].index(True)
        ret_stmt = body_stmts[ret_idx]
        if isinstance(ret_stmt.value, ast.Name):
            ret_name = ret_stmt.value.id
        else:
            raise NotImplementedError("No expressions in return stmt, for now")
        return ret_name

    def __call__(self, table):
        result = self.func(table)
        if isinstance(table, _Table):
            func_signature = self.name + '.' + str(table)
            if func_signature not in self.udf_table:
                code = self.emit_code(table, result)
                self.udf_table[func_signature] = code
        return result

    def emit_code(self, table, result):
        input_types = table.types
        input_types = [SQLTYPES[it] for it in input_types]
        input_types = ", ".join([f"i{i} {it}" for i, it in enumerate(input_types)])
        output_type = SQLTYPES[self._get_output_type()]
        ret_name = self._get_return_name()
        if isinstance(self.output_type, _Table):
            ncols_out = result.shape[1]
            output_types = ", ".join(
                [f"{ret_name}_{i} {output_type}" for i in range(ncols_out)]
            )
            output_types = "Table(" + output_types + ")"
        elif issubclass(self.output_type, Number):
            output_types = output_type
        code = [f"CREATE OR REPLACE FUNCTION {self.name}({input_types})"]
        code.append(f"RETURNS {output_types}")
        code.append("LANGUAGE PYTHON {")
        code.append("    import numpy as np")
        code.append("    from arraybundle import ArrayBundle")
        code.append("    table = ArrayBundle(_columns)\n")
        code.append("    # start method body")
        if isinstance(self.output_type, _Table):
            code.extend(self.func_bodies[self.name][:-1])
            code.append("    _colrange = range(table.shape[1])")
            code.append(f'    _names = (f"{ret_name}_{{i}}" for i in _colrange)')
            line = f"    _result = {{n: c for n, c in zip(_names, {ret_name})}}"
            code.append(line)
            code.append("    return _result")
        elif issubclass(self.output_type, Number):
            code.extend(self.func_bodies[self.name])
        code.append("};")
        return "\n".join(code)


def parse_func(func):
    sourcelines = inspect.getsourcelines(func)[0]
    declaration = sourcelines[1]
    body = sourcelines[2:]  # remove declaration TODO better solution with regex
    return declaration, body


def verify_annotations(func):
    """Verifies that func is well annotated. All algorithm methods should be
    fully annotated (parameters and return val) in order to become UDFs.
    """
    parameters = inspect.signature(func).parameters
    annotations = func.__annotations__
    parameters = dict(parameters)
    try:  # UDFs can be methods or simple functions
        del parameters["self"]
    except KeyError:
        pass
    if len(parameters) != len(annotations) - 1:  # subtract return annotation
        msg = "This method should be fully annotated. "
        msg += f"Some annotations are missing: {annotations}"
        raise SyntaxError(msg)


def udf_wrapper(func):
    wrapper = UDFFactory(func)
    return wrapper


udf = udf_wrapper  # create simpler alias


# -------------------------------------------------------- #
# Table                                                    #
# -------------------------------------------------------- #
# Table kinds:
TYPE_ANNOTATOR = 1   # case 1 0 0
MOCK_ARRAY = 2       # case 1 1 0
ARRAY = 3            # case 0 0 1


class _Table(np.lib.mixins.NDArrayOperatorsMixin):
    """
    Table grammar:
    --------------
        start := 'Table[' KEY ']'

        key := TYPE | TUPLE | SLICE

        TYPE := 'int'
             | 'float'
             | 'str'

        TUPLE := LIST_OF_TYPES
              | TYPE ',' '...'

        LIST_OF_TYPES := TYPE ',' LIST_OF_TYPES

        SLICE := TYPE ':' NUMBER

    Examples:
    ---------
        Table[int]              -> types=(int,), ncols=1
        Table[int, float]       -> types=(int, float), ncols=2
        Table[int:4]            -> types=(int,), ncols=4
        Table[int, ...]         -> types=(int,), ncols=None

    Not supported yet:
        Table[int, float, ...]       : one int and any number of floats
        Table[..., int, float]       : any number of ints and one float
        Table[int:SAME, float:SAME]  : any number of ints and the same number of floats
    """
    def __init__(self, types=None, *, ncols=None, array=None):
        if types:
            self.types = types
            if (ncols, array) == (None, None):  # case 1 0 0
                self.kind = TYPE_ANNOTATOR
            elif ncols and array is None:  # case 1 1 0
                self.kind = MOCK_ARRAY
                self.shape = (2, ncols)
                if all(tp == int for tp in types):
                    self.array = np.random.randint(50, size=self.shape)
                elif all(tp == float for tp in types):
                    self.array = np.random.rand(*self.shape)
                elif all(tp == str for tp in types):
                    self.array = random.choices(['a', 'b', 'c'], k=2*ncols)
                    self.array = np.array(self.array).reshape(2, ncols)
                else:  # catch all case TODO replace with more choices
                    self.array = np.random.rand(*self.shape)
        elif (types, ncols) == (None, None) and array is not None:  # case 0 0 1
            self.kind = ARRAY
            self.types = (array.dtype,) * array.shape[1]
            self.shape = array.shape
            self.array = array
        else:
            msg = "This class can only be instantiated in one of four ways, "
            msg += "_Table(), _Table(TYPE), _Table(TYPE, NCOLS) or _Table(array=ARRAY)"
            raise ValueError(msg)

    def __getitem__(self, key):
        if isinstance(key, Number) or isinstance(key, slice):
            raise KeyError("You shouldn't be accessing rows of Table object")
        if self.kind in (MOCK_ARRAY, ARRAY):
            value = self.array[key]
            if isinstance(value, np.ndarray):
                value = value[:, np.newaxis] if len(value.shape) == 1 else value
                result = Table(value)
                result.kind = self.kind
                return result
            elif isinstance(value, Number):
                return value
        else:
            raise SyntaxError('Cannot get item of TYPE_ANNOTATOR')

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        kinds = [input_.kind for input_ in inputs if isinstance(input_, type(self))]
        if len(kinds) == 1:
            out_kind = kinds[0]
        elif kinds == [MOCK_ARRAY, MOCK_ARRAY]:
            out_kind = MOCK_ARRAY
        elif kinds == [ARRAY, ARRAY]:
            out_kind = ARRAY
        elif set(kinds) == {MOCK_ARRAY, ARRAY}:
            out_kind = ARRAY
        inputs = [
            input_ if not isinstance(input_, type(self)) else input_.array
            for input_ in inputs
        ]
        output = getattr(ufunc, method)(*inputs, **kwargs)
        if isinstance(output, Number):
            return output
        elif isinstance(output, np.ndarray):
            new_table = Table(array=output)
            new_table.kind = out_kind
            return new_table

    def __repr__(self):
        cls_name = 'Table'
        if self.kind == TYPE_ANNOTATOR:
            return f'{cls_name}[{self.types[0].__name__}, ...]'
        elif self.kind == MOCK_ARRAY:
            try:
                tp_names = [tp.__name__ for tp in self.types]
            except AttributeError:
                tp_names = [tp.name for tp in self.types]
            return f'{cls_name}[{",".join(tp_names)}]'
        elif self.kind == ARRAY:
            array = repr(self.array).split('\n')
            array = '\n'.join(['    ' + ln for ln in array])
            return f'{cls_name}(\n' + array + '\n)'

    def __len__(self):
        return self.shape[0]

    @property
    def transpose(self):
        new_table = _Table(array=self.array.T)
        new_table.kind = self.kind
        return new_table

    T = transpose


class TableFactory:
    def __getitem__(self, types):
        ALLOWED_TYPES = (int, float, str)
        if types in ALLOWED_TYPES:
            return _Table((types,), ncols=1)
        elif isinstance(types, tuple):
            if len(types) == 2 and types[1] == Ellipsis:
                return _Table((types[0],))
            elif all(tp in ALLOWED_TYPES for tp in types):
                return _Table(types, ncols=len(types))
            else:
                raise NotImplementedError('No yet')
        elif isinstance(types, slice):
            tp = types.start
            ncols = types.stop
            return _Table((tp,) * ncols, ncols=ncols)
        else:
            raise NotImplementedError('Not yet')

    def __call__(self, array):
        return _Table(array=array)


Table = TableFactory()


# -------------------------------------------------------- #
# Examples                                                 #
# -------------------------------------------------------- #
@udf
def compute_gramian(table: Table[float, ...]) -> Table[float, ...]:
    gramian = table.T @ table
    return gramian


@udf
def sum_of_squares(table: Table[float, ...]) -> float:
    sos = np.sum(table * table)
    return sos


@udf
def half_table(table: Table[int, ...]) -> Table[int, ...]:
    ncols = table.shape[1]
    if ncols >= 2:
        result = table[:, 0:ncols // 2]
    else:
        result = table
    return result


print(compute_gramian(Table(np.array([[1, 2, 3], [10, 20, 30]]))))
print(compute_gramian(np.array([[1, 2, 3], [10, 20, 30]])))
print(sum_of_squares(Table[float:4]))
print(half_table(Table[int:6]))
