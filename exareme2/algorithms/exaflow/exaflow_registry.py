from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from exareme2.utils import Singleton


class ExaflowRegistry(metaclass=Singleton):
    def __init__(self) -> None:
        self._registry: Dict[str, Callable] = {}

    # ------------------------------------------------------------------ #
    def _makeRegistry_key(self, func: Callable) -> str:
        """module basename + '_' + function name"""
        return f"{func.__module__.split('.')[-1]}_{func.__name__}"

    # ------------------------------------------------------------------ #
    def register(self, func: Callable) -> Callable:
        key = self._make_key(func)
        if key in self._registry:
            # already registered – only complain if somebody tries
            # to hijack the key with a *different* callable
            if self._registry[key] is not func:
                raise ValueError(
                    f"Duplicate UDF key detected: {key} "
                    "(existing entry refers to a different function)"
                )
            return func  # ← quietly accept the re-registration
        self._registry[key] = func
        return func

    # ------------------------------------------------------------------ #
    def resolve_key(self, item: Union[str, Callable]) -> str:
        if callable(item):
            key = self._make_key(item)
            if key not in self._registry:  # first sighting → register
                self._registry[key] = item
            return key

    # conveniences ------------------------------------------------------- #
    def get_udf(self, key: str) -> Optional[Callable]:
        return self._registry.get(key)

    def get_available_udfs(self) -> List[str]:
        return list(self._registry.keys())


# Singleton instance
exaflow_registry = ExaflowRegistry()


# Decorator to simplify registration
def exaflow_udf(func):
    return exaflow_registry.register(func)
