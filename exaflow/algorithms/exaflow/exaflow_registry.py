from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Callable
from typing import Dict

from exaflow.utils import Singleton

"""
Lightweight registry for exaflow UDFs.

- UDFs are registered with a stable key derived from `func.__qualname__` and
  the defining module (see `get_udf_registry_key`).
- Set `with_aggregation_server=True` when a UDF needs an aggregation server
  session injected via `agg_client` (see logistic_regression for an example).
"""


def _hash(text: str) -> str:
    return base64.b32encode(hashlib.sha1(text.encode()).digest())[:4].decode().lower()


def get_udf_registry_key(func: Callable) -> str:
    module_part = func.__module__.split(".")[-1]
    qual = func.__qualname__.replace(".", "__")
    return f"{qual}_{_hash(module_part)}"


@dataclass(frozen=True)
class UDFInfo:
    func: "Callable"
    with_aggregation_server: "bool"


class ExaflowRegistry(metaclass=Singleton):
    def __init__(self) -> None:
        self._registry: Dict[str, UDFInfo] = {}

    def register(self, func: Callable, *, with_aggregation_server: bool = False) -> str:
        """
        Register a UDF and return its registry key. Raises if the same key is
        reused for a different function to avoid ambiguity.
        """
        key = get_udf_registry_key(func)
        if key in self._registry and self._registry[key].func is not func:
            raise ValueError(f"Duplicate registration for key {key!r}")
        self._registry[key] = UDFInfo(func, with_aggregation_server)
        return key

    def aggregation_server_required(self, key: str) -> bool:
        info = self._registry.get(key)
        return bool(info and info.with_aggregation_server)

    def get_func(self, key: str) -> Callable:
        info = self._registry.get(key)
        return info.func

    def __repr__(self) -> str:
        return f"<ExaflowRegistry {len(self._registry)} entries>"


exaflow_registry = ExaflowRegistry()


def exaflow_udf(
    _func: Callable | None = None, *, with_aggregation_server: bool = False
):
    def decorator(func: Callable) -> Callable:
        exaflow_registry.register(func, with_aggregation_server=with_aggregation_server)
        return func

    if _func is None:
        return decorator

    return decorator(_func)
