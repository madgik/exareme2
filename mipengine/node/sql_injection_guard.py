import inspect
import re
from functools import wraps
from inspect import getfullargspec


def get_arg_value_from_args_and_kwargs(func, param_name, args, kwargs):
    arglist = getfullargspec(func)
    if kwargs.get(param_name):
        return kwargs.get(param_name)
    else:
        param_index = arglist.args.index(param_name)
        if param_index < len(args):
            return args[param_index]

    return None


def _validate_list(validation_func, list):
    for elem in list:
        _validate_strings(validation_func, elem)


def _validate_dict(validation_func, dict):
    for key in dict.keys():
        _validate_strings(validation_func, key)
    for elem in dict.values():
        _validate_strings(validation_func, elem)


def _validate_strings(validation_func, value):
    """
    Validates, using the validation_func, all the str elements.
    Iterates through lists and dicts.
    """
    if isinstance(value, str):
        validation_func(value)
    elif isinstance(value, list):
        _validate_list(validation_func, value)
    elif isinstance(value, dict):
        _validate_dict(validation_func, value)


def _validate_param_exists_in_function(func, param_name):
    arglist = getfullargspec(func)
    if param_name not in arglist.args:
        raise ValueError(
            f"Function '{func.__name__}' has no argument named '{param_name}'."
        )


def sql_injection_guard(param_name: str, validation_func, optional=False):
    """
    Validates, using the validation_func, the argument passed in the method.
    Validates all strings in lists or dicts.
    Validates both posargs and kwargs.
    """

    def sql_injection_guard_internal(func):
        _validate_param_exists_in_function(func, param_name)

        @wraps(func)
        def wrapper(*args, **kwargs):
            arg_value = get_arg_value_from_args_and_kwargs(
                func, param_name, args, kwargs
            )
            if not arg_value and not optional:
                raise ValueError(
                    f"Parameter '{param_name}' was not provided, thus it couldn't be validated."
                )

            try:
                _validate_strings(validation_func, arg_value)
            except Exception as exc:
                raise ValueError(
                    f"A validation error occurred with parameter: '{param_name}'. Value: '{arg_value}'. Exception type: {type(exc)}. Exception: {exc}"
                )
            return func(*args, **kwargs)

        # Copy the func signature to the wrapper, so you can later be able to check the argspec.
        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return sql_injection_guard_internal


def isalnum(string):
    if not string.isalnum():
        raise ValueError(f"The value provided: '{string}' is not alphanumeric.")


def isalpha(string):
    if not string.isalpha():
        raise ValueError(f"The value provided: '{string}' is not alphabetic.")


def isidentifier(string):
    if not string.isidentifier():
        raise ValueError(f"The value provided: '{string}' is not an identifier.")


def is_hyphen_identifier(string):
    # Regex that allows letters, digits, hyphen and underscore.
    hyphen_identifier_regex = "^([A-Za-z0-9_-])+$"
    if not re.match(hyphen_identifier_regex, string):
        raise ValueError(
            f"The value provided: '{string}' is not a hyphen identifier (Letters, digits, underscore and hyphen allowed)."
        )


def is_socket_address(string):
    # Regex that allows an ip or a hostname and then a port
    socket_address_regex = "^([0-9]{1,3}(?:\.[0-9]{1,3}){3}|[a-zA-Z]+):([0-9]{1,5})$"
    if not re.match(socket_address_regex, string):
        raise ValueError(f"The value provided: '{string}' is not a socket address.")


data_model_regex = "([A-Za-z0-9_])+:([A-Za-z0-9]|\.)+"


def isdatamodel(string):
    # Regex that allows an alnum string then ':' and then an alnum string with possible dots as well
    if not re.match(f"^{data_model_regex}$", string):
        raise ValueError(f"The value provided: '{string}' is not a data model label.")


def is_primary_data_table(string):
    # If the string matches an identifier, no further validation needs to happen.
    if string.isidentifier():
        return

    # Regex that allows a data model primary data table name
    primary_data_table_regex = f'^"{data_model_regex}"\."([A-Za-z0-9_])+"$'
    if not re.match(primary_data_table_regex, string):
        raise ValueError(f"The value provided: '{string}' is not a primary data table.")
