import re
from functools import wraps


def validate_identifier_names(func):
    wraps(func)

    def wrapper(*args, **kwargs):
        all_args = list(args) + list(kwargs.values())
        for arg in all_args:
            if type(arg) == str:
                if not arg.isidentifier() and not check_proper_url_format(arg):
                    raise ValueError(f"Not allowed character in argument: {arg}!")
            elif type(arg) == list:
                for item in arg:
                    if not item.isidentifier() and not check_proper_url_format(item):
                        raise ValueError(f"Not allowed character in argument: {item}!")
        return func(*args, **kwargs)

    return wrapper


def check_proper_url_format(url: str):
    regex_between_0_to_255 = "([01]?[0-9]?[0-9]|2[0-4][0-9]|25[0-5])"
    regex_any_alphanumeric = "([a-zA-Z0-9]*)"
    # The format of the URL of a REMOTE TABLE is: mapi:monetdb://<host>:<port>/<dbname>
    return re.search(
        f"^mapi:monetdb://{regex_between_0_to_255}\.{regex_between_0_to_255}\.{regex_between_0_to_255}\.{regex_between_0_to_255}:{regex_any_alphanumeric}/{regex_any_alphanumeric}$",
        url)
