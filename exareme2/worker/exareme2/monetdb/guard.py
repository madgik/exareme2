import inspect
import re
from functools import wraps
from inspect import FullArgSpec
from numbers import Number
from typing import Any
from typing import Callable
from typing import Optional

from exareme2.worker_communication import SMPCTablesInfo
from exareme2.worker_communication import TableInfo
from exareme2.worker_communication import TableSchema
from exareme2.worker_communication import WorkerLiteralDTO
from exareme2.worker_communication import WorkerSMPCDTO
from exareme2.worker_communication import WorkerTableDTO
from exareme2.worker_communication import WorkerUDFDTO


def sql_injection_guard(**validators: Optional[Callable[[Any], bool]]):
    """
    Parametrized  decorator guarding SQL generating functions from strings
    suspect for SQL injections.

    All  functions  that receive string arguments, sent from Controller to
    Worker  via  a  Celery  task,  should  be  decorated  with the decorator
    returned   by  this  function.  This  function  accepts  only  keyword
    arguments  which  should  be  one-to-one  with  the  arguments  of the
    decorated  function.  For  each argument there should be one validator
    function  or  None  for  arguments  which  don't  need validation. The
    validator  function  is  a  predicate:  it receives a single value and
    returns a bool.

    Parameters
    ----------
    validators : Optional[Callable[[Any], bool]]
        Validator functions, one for each argument of the decorated function

    Examples
    --------
    >>> @sql_injection_guard(a=str.isalnum, b=my_validator, c=None)
    ... def f(a, b, c):
    ...     ...
    """

    def decorator(func):
        argspec = inspect.getfullargspec(func)
        if set(argspec.args) != set(validators.keys()):
            raise ValueError(
                "sql_injection_guard validators do not match function "
                f"{func.__name__} args.\n"
                f"Args mismatch: {set(argspec.args) - set(validators.keys())}.\n"
                "Make sure to add one validator per function arg, passing "
                "None for args that don't need validation."
            )

        @wraps(func)
        def wrapper(*args, **kwargs):
            values = get_arg_values(argspec, args, kwargs)
            validate_arg_values(validators, values)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_arg_values(argspec: FullArgSpec, args: list, kwargs: dict) -> dict:
    arg_values = get_named_defaults(argspec)
    arg_values.update(get_named_posarg_values(argspec, args))
    arg_values.update(kwargs)
    return arg_values


def get_named_defaults(argspec: FullArgSpec) -> dict:
    if argspec.defaults:
        return dict(zip(reversed(argspec.args), reversed(argspec.defaults)))
    return {}


def get_named_posarg_values(argspec: FullArgSpec, args: list) -> dict:
    return dict(zip(argspec.args, args))


def validate_arg_values(validators: dict, arg_values: dict) -> None:
    for arg, value in arg_values.items():
        validator = validators[arg] or (lambda x: True)
        if not validator(value):
            raise InvalidSQLParameter(
                f"Arg '{arg}' with value '{value}' cannot be validated "
                f"with validator '{validator.__name__}'."
            )


class InvalidSQLParameter(Exception):
    pass


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Validators ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# ip regex
#   https://stackoverflow.com/questions/5284147/validating-ipv4-addresses-with-regexp
# hostname regex (RFC 952)
#   https://stackoverflow.com/questions/106179/regular-expression-to-match-dns-hostname-or-ip-address
ip_re = r"((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.?\b){4}"
alpha_re = r"[a-zA-Z]"
alnum_re = r"[a-zA-Z0-9]"
alnumhyphen_re = r"[a-zA-Z0-9-]"
hostname_re = rf"""(({alpha_re}|{alpha_re}{alnumhyphen_re}*{alnum_re})\.)*
                    ({alpha_re}|{alpha_re}{alnumhyphen_re}*{alnum_re})"""
port_re = r"(?P<port>\d{1,5})"
socketaddress_ptrn = re.compile(rf"(?:{ip_re}|{hostname_re}):{port_re}", re.A | re.X)
version_re = r"\w+(\.\w)*"
datamodel_re = rf"\w+:{version_re}"
datamodel_ptrn = re.compile(datamodel_re, re.A)
sqlidentifier_re = r"[a-z_][a-z0-9_]*"
datatable_ptrn = re.compile(rf'"{datamodel_re}"\.("\w+"|{sqlidentifier_re})', re.A)
uuid_re = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
uuid_ptrn = re.compile(uuid_re, re.A | re.I)


def is_socket_address(string):
    match = socketaddress_ptrn.fullmatch(string)
    if not match:
        return False
    port = int(match.group("port"))
    if 1 <= port <= 65535:
        return True
    return False


def is_datamodel(string):
    return bool(datamodel_ptrn.fullmatch(string))


def is_primary_data_table(string):
    return string.isidentifier() or bool(datatable_ptrn.fullmatch(string))


def is_list_of_identifiers(lst):
    return True


def is_valid_filter(filter):
    if filter is None:
        return True
    if "id" in filter:  # base case
        return filter["id"].isidentifier()
    if "rules" in filter:  # recursive case
        return all(is_valid_filter(rule) for rule in filter["rules"])
    return False


def is_valid_table_schema(schema: TableSchema):
    return all(col.name.isidentifier() for col in schema.columns)


def is_valid_udf_arg(arg):
    if isinstance(arg, WorkerUDFDTO):
        if isinstance(arg, WorkerTableDTO):
            return is_valid_table_info(arg.value)
        elif isinstance(arg, WorkerSMPCDTO):
            return is_valid_smpc_tables_info(arg.value)
        elif isinstance(arg, WorkerLiteralDTO):
            return is_valid_literal_value(arg.value)
        raise NotImplementedError(f"{arg.__class__} has no validator implementation")
    raise TypeError("UDF args have to be subclasses of WorkerUDFDTO")


def is_valid_table_info(info: TableInfo):
    return info.name.isidentifier() and is_valid_table_schema(info.schema_)


def is_valid_smpc_tables_info(info: SMPCTablesInfo):
    return (
        is_valid_table_info(info.template)
        and (is_valid_table_info(info.sum_op) if info.sum_op else True)
        and (is_valid_table_info(info.min_op) if info.min_op else True)
        and (is_valid_table_info(info.max_op) if info.max_op else True)
    )


def is_valid_literal_value(val):
    if isinstance(val, Number):
        return True
    if isinstance(val, str):
        # The  MonetDB  parser  has  a weird behaviour when parsing python UDFs.
        # When  the  sequence '};' appears in a string in the python code, it is
        # interpreted  as  the  end  of  the  UDF  definition. Weirdly, when the
        # sequence is part of the actual code and not inside a string, it is not
        # interpreted  like  that. This behaviour can be exploited to perform an
        # SQL injection attack, passing for example the string '};DROP TABLE x;'
        # as  a  literal.  To  avoid  that,  this  sequence  is  prohibited from
        # appearing in a literal string.
        prohibited_sequence = "};"
        return prohibited_sequence not in val
    if isinstance(val, list):
        return all(is_valid_literal_value(elem) for elem in val)
    if isinstance(val, dict):
        return all(
            is_valid_literal_value(k) and is_valid_literal_value(v)
            for k, v in val.items()
        )
    raise NotImplementedError(
        "Currently literal values can be of types str, Number, list or dict,\n"
        "and nestings of the above."
    )


def udf_posargs_validator(posargs):
    return all(is_valid_udf_arg(arg) for arg in posargs.args)


def udf_kwargs_validator(kwargs):
    return all(is_valid_udf_arg(arg) for _, arg in kwargs.args.items())


def output_schema_validator(schema):
    return schema is None or all(name.isidentifier() for name, _ in schema)


def is_valid_request_id(string):
    return string.isalnum() or bool(uuid_ptrn.fullmatch(string))
