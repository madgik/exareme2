import os
from importlib.resources import open_binary
from types import SimpleNamespace

import tomli

# sane defaults if env is unset
_DEFAULTS = {
    "host": "0.0.0.0",
    "port": 50051,
    "max_workers": 10,
    "timeout": 60,
    "log_level": "INFO",
}


def _load_config():
    # 1) load raw dict
    cfg_path = os.getenv("AGG_SERVER_CONFIG_FILE")
    if cfg_path:
        with open(cfg_path, "rb") as f:
            raw = tomli.load(f)
    else:
        with open_binary(__package__, "config.toml") as f:
            raw = tomli.load(f)

    # 2) substitute and coerce
    cfg = {}
    for key, val in raw.items():
        if isinstance(val, str) and val.startswith("$"):
            # strip leading '$' and lookup
            env_name = val[1:]
            val = os.getenv(env_name, _DEFAULTS[key])
        # convert numbers
        if key in ("port", "max_workers", "timeout"):
            val = int(val)
        cfg[key] = val

    return SimpleNamespace(**cfg)


config = _load_config()


from .aggregation_server_pb2 import *
from .aggregation_server_pb2_grpc import *

# ——— Expose the “public” API ———
from .constants import AggregationType
from .server import serve

__all__ = [
    "config",
    "AggregationType",
    # all your generated message & stub names from pb2 / pb2_grpc
    *[name for name in dir() if name.endswith("Request") or name.endswith("Response")],
    "serve",
]
