from collections import UserDict
from functools import wraps

__all__ = ["dispatch"]


def dispatch(func):
    """Decorator for multiple dispatch functions. First decorate the function
    prototype and then use register method of class Dispatcher to further
    decorate function implementations for different input types. This
    particular implementation expects the first argument to be a tuple of other
    types Then the dispatch mechanism only concerns the types in this tuple,
    ignoring all remaining function arguments.

    Examples:
    ---------
        >>> @dispatch
        ... def func(x, *args, **kwargs):
        ...     raise NotImplementedError
        ...
        >>> @func.register(int)
        ... def _(x, *args, **kwargs):
        ...     print('x is a tuple of one int')
        ...
        >>> @func.register(int, float)
        ... def _(x, *args, **kwargs):
        ...     print('x is a tuple of an int and a float')
        ...
        >>> func((4,))
        x is a tuple of one int
        >>> func((1, 3.14), hello='hi')
        x is a tuple of an int and a float
    """
    dispatcher = Dispatcher(func)
    return dispatcher


class Dispatcher:
    def __init__(self, prototype):
        self._prototype = prototype
        self.registry = TypeRegistry(default=self._prototype)

    def __call__(self, first_args, *args, **kwargs):
        types = tuple(type(arg) for arg in first_args)
        return self.registry[types](first_args, *args, **kwargs)

    def register(self, *types):
        def outter(func):
            self.registry[types] = func

            @wraps(func)
            def inner(*args, **kwargs):
                return func(*args, **kwargs)

            return inner

        return outter


class TypeRegistry(UserDict):
    def __init__(self, default):
        self.default = default
        super().__init__()

    def __getitem__(self, t):
        if not isinstance(t, tuple):
            raise KeyError(
                f"TypeRegistry expecting tuple of types as key. Got {type(t)} instead."
            )
        types = list(self.keys())
        match_keys = [self._are_subclasses(t, registered_t) for registered_t in types]
        if True in match_keys:
            found_idx = match_keys.index(True)
            return self.data[types[found_idx]]
        else:
            return self.default

    @staticmethod
    def _are_subclasses(children, parents):
        if len(children) != len(parents):
            return False
        return all(
            issubclass(child, parent) for child, parent, in zip(children, parents)
        )
