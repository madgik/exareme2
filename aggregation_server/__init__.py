import os
from importlib.resources import open_binary
from types import SimpleNamespace

# --- third-party ---
import tomli

# --------------------------------------------------------------------------
# 1. define defaults and load configuration *first*
# --------------------------------------------------------------------------

_DEFAULTS = {
    "port": 50051,
    "max_grpc_connections": 10,
    "max_wait_for_aggregation_inputs": 60,
    "log_level": "INFO",
}


def _load_config():
    cfg_path = os.getenv("AGG_SERVER_CONFIG_FILE")
    if cfg_path:
        with open(cfg_path, "rb") as f:
            raw = tomli.load(f)
    else:
        with open_binary(__package__, "config.toml") as f:
            raw = tomli.load(f)

    cfg = {}
    for key, val in raw.items():
        if isinstance(val, str) and val.startswith("$"):
            env_name = val[1:]
            val = os.getenv(env_name, _DEFAULTS[key])
        if key in ("port", "max_grpc_connections", "max_wait_for_aggregation_inputs"):
            val = int(val)
        cfg[key] = val

    return SimpleNamespace(**cfg)


config = _load_config()

from .aggregation_server_pb2 import *
from .aggregation_server_pb2_grpc import *
from .constants import AggregationType
from .server import serve

__all__ = [
    "config",
    "AggregationType",
    *[name for name in dir() if name.endswith("Request") or name.endswith("Response")],
    "serve",
]
