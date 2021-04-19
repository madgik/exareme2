import re
from functools import wraps


def sql_injection_guard(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        all_args = list(args) + list(kwargs.values())
        validate_argument(all_args)
        return func(*args, **kwargs)

    return wrapper


def validate_argument(argument):
    for arg in argument:
        if isinstance(arg, str):
            if arg.isidentifier():
                continue
            elif arg.isalnum():
                continue
            elif has_proper_db_socket_address_format(arg):
                continue
            raise ValueError(f"Not allowed character in argument: {arg}")
        elif isinstance(arg, list):
            validate_argument(arg)
        elif isinstance(arg, dict):
            validate_argument(arg)


def has_proper_db_socket_address_format(db_socket_address: str):
    db_socket_address_match = re.fullmatch(
        r"""
        (\d{0,3})
        \.
        (\d{0,3})
        \.
        (\d{0,3})
        \.
        (\d{0,3})
        :
        (\d{0,5})
        """,
        db_socket_address,
        re.VERBOSE,
    )
    return db_socket_address_match is not None
