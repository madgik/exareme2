from functools import wraps
import re


def validate_sql_params(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        all_args = list(args) + list(kwargs.values()) + list(kwargs.keys())
        for arg in all_args:
            if isinstance(arg, str):
                _validate_sql_param(arg)
            elif isinstance(arg, list):
                wrapper(*arg)
            elif isinstance(arg, dict):
                wrapper(**arg)
            else:
                raise ValueError(f"Expected valid SQL parameter, got {arg}")
        return func(*args, **kwargs)

    return wrapper


def _validate_sql_param(arg):
    if arg.isidentifier():
        return
    elif arg.isalnum():
        return
    elif _validate_socket_address(arg):
        return
    else:
        raise ValueError(f"Invalid SQL parameter {arg}")


def _validate_socket_address(arg):
    return re.fullmatch(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}", arg) is not None
