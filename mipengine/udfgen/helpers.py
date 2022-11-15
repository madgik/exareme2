import ast
import base64
import hashlib
import inspect
import re
from textwrap import dedent
from typing import List
from typing import Tuple

SEP = ","


def iotype_to_sql_schema(iotype, name_prefix=""):
    column_names = iotype.column_names(name_prefix)
    types = [dtype.to_sql() for _, dtype in iotype.schema]
    sql_params = [f'"{name}" {dtype}' for name, dtype in zip(column_names, types)]
    return SEP.join(sql_params)


def recursive_repr(self: object) -> str:
    """Recursively representing an object using its attribute reprs."""
    cls = type(self).__name__
    attrs = self.__dict__
    publicattrs = {
        name: attr for name, attr in attrs.items() if not name.startswith("_")
    }
    attrs_repr = ",".join(name + "=" + repr(attr) for name, attr in publicattrs.items())
    return f"{cls}({attrs_repr})"


def remove_empty_lines(lines: List[str]):
    return [
        remove_trailing_newline(line)
        for line in lines
        if not re.fullmatch(r"^$", line.strip())
    ]


def remove_trailing_newline(string):
    return re.sub(r"\n$", "", string)


def parse_func(func):
    """Get function AST"""
    code = dedent(inspect.getsource(func))
    return ast.parse(code)


def get_func_body_from_ast(tree):
    assert len(tree.body) == 1
    funcdef, *_ = tree.body
    return funcdef.body


def get_return_names_from_body(statements) -> Tuple[str, List[str]]:
    """Returns names of variables in return statement. Assumes that a return
    statement exists and is of type ast.Name or ast.Tuple because the validation is
    supposed to happen before (in validate_func_as_udf)."""
    ret_stmt = next(s for s in statements if isinstance(s, ast.Return))
    if isinstance(ret_stmt.value, ast.Name):
        return ret_stmt.value.id, []  # type: ignore
    elif isinstance(ret_stmt.value, ast.Tuple):
        main_ret = ret_stmt.value.elts[0].id
        sec_rets = [value.id for value in ret_stmt.value.elts[1:]]
        return main_ret, sec_rets
    else:
        raise NotImplementedError


def make_unique_func_name(func) -> str:
    """Creates a unique function name composed of the function name, an
    underscore and the module's name hashed, encoded in base32 and truncated at
    4 chars."""
    full_module_name = func.__module__
    module_name = full_module_name.split(".")[-1]
    hash_ = get_base32_hash(module_name)
    return func.__name__ + "_" + hash_.lower()


def get_base32_hash(string, chars=4):
    hash_ = hashlib.sha256(string.encode("utf-8")).digest()
    hash_ = base64.b32encode(hash_).decode()[:chars]
    return hash_


def get_func_parameter_names(func):
    """Gets the list of parameter names of a function."""
    signature = inspect.signature(func)
    return list(signature.parameters.keys())


def mapping_inverse(mapping):
    """Inverses mapping if it is bijective or raises error if not."""
    if len(set(mapping.keys())) != len(set(mapping.values())):
        raise ValueError(f"Mapping {mapping} cannot be reversed, it is not bijective.")
    return dict(zip(mapping.values(), mapping.keys()))


def compose_mappings(map1, map2):
    """Returns f[x] = map2[map1[x]], or using Haskell's dot notation
    map1 .  map2, if mappings are composable. Raises a ValueError otherwise."""
    if not set(map1.values()) <= set(map2.keys()):
        raise ValueError(f"Mappings are not composable, {map1}, {map2}")
    return {key: map2[val] for key, val in map1.items()}


def merge_mappings_consistently(mappings: List[dict]) -> dict:
    """Merges a list of mappings into a single mapping, raising a ValueError if
    mappings do not coincide."""
    merged = dict()
    for mapping in mappings:
        if not mappings_coincide(merged, mapping):
            raise ValueError(f"Cannot merge inconsistent mappings: {merged}, {mapping}")
        merged.update(mapping)
    return merged


def mappings_coincide(map1: dict, map2: dict) -> bool:
    """Returns True if the image of the intersection of the two mappings is
    the same, False otherwise. In other words, True if the mappings coincide on
    the intersection of their keys."""
    intersection = set(map1.keys()) & set(map2.keys())
    if any(map1[key] != map2[key] for key in intersection):
        return False
    return True


def merge_args_and_kwargs(param_names, args, kwargs):
    """Merges args and kwargs for a given list of parameter names into a single
    dictionary."""
    merged = dict(zip(param_names, args))
    merged.update(kwargs)
    return merged


def get_items_of_type(type_, mapping):
    """Gets items of mapping being instances of a given type."""
    return {key: val for key, val in mapping.items() if isinstance(val, type_)}


def is_any_element_of_type(type_, elements):
    return any(isinstance(elm, type_) for elm in elements)
