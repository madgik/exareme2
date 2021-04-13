import re
from functools import wraps


def sql_injection_guard(func):
    wraps(func)

    def wrapper(*args, **kwargs):
        all_args = list(args) + list(kwargs.values())
        for arg in all_args:
            if type(arg) == str:
                if (
                    not arg.isidentifier()
                    and not arg.isalnum()
                    and not has_proper_db_location_format(arg)
                ):
                    raise ValueError(f"Not allowed character in argument: {arg}")
            elif type(arg) == list:
                for item in arg:
                    if (
                        not item.isidentifier()
                        and not item.isalnum()
                        and not has_proper_db_location_format(item)
                    ):
                        raise ValueError(f"Not allowed character in argument: {item}")
        return func(*args, **kwargs)

    return wrapper


def has_proper_db_location_format(db_location: str):
    regex_between_0_to_255 = "([01]?[0-9]?[0-9]|2[0-4][0-9]|25[0-5])"
    regex_any_alphanumeric = "([a-zA-Z0-9]*)"
    ip_regex = (
        f"{regex_between_0_to_255}"
        f"\.{regex_between_0_to_255}"
        f"\.{regex_between_0_to_255}"
        f"\.{regex_between_0_to_255}"
    )
    # The format of the db location of a REMOTE TABLE is: <host>:<port>
    url_match = re.match(f"{ip_regex}:{regex_any_alphanumeric}$", db_location)
    return url_match is not None
