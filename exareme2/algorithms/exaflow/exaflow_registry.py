from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Callable
from typing import Dict

from exareme2.utils import Singleton


def _hash(text: str) -> str:
    return base64.b32encode(hashlib.sha1(text.encode()).digest())[:4].decode().lower()


def get_udf_registry_key(func: Callable) -> str:
    module_name = func.__module__
    module_part = module_name.split(".")[-1]
    qual = func.__qualname__.replace(".", "__")
    return f"{module_part}__{qual}_{_hash(module_name)}"


@dataclass(frozen=True)
class UDFInfo:
    func: "Callable"
    with_aggregation_server: "bool"


class ExaflowRegistry(metaclass=Singleton):
    def __init__(self) -> None:
        self._registry: Dict[str, UDFInfo] = {}

    def register(self, func: Callable, *, with_aggregation_server: bool = False) -> str:
        key = get_udf_registry_key(func)
        existing = self._registry.get(key)

        if existing is not None:
            if existing.func is func:
                return key

            same_definition = (
                existing.func.__module__ == func.__module__
                and existing.func.__qualname__ == func.__qualname__
            )

            if not same_definition:
                raise ValueError(f"Duplicate registration for key {key!r}")

            # Module reloads recreate function objects, so refresh the registry entry.

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
