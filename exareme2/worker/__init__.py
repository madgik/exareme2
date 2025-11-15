import os
from importlib.resources import open_text
from pathlib import Path

import envtoml

from exareme2 import AttrDict
from exareme2 import worker


def _ensure_duckdb_config(cfg: AttrDict) -> None:
    duckdb_cfg = getattr(cfg, "duckdb", AttrDict({}))
    duckdb_path = duckdb_cfg.get("path")
    if not duckdb_path:
        default_path = Path(cfg.data_path) / f"{cfg.identifier}.duckdb"
        duckdb_path = str(default_path)
    duckdb_cfg.path = duckdb_path
    cfg.duckdb = duckdb_cfg


def _ensure_data_loader_config(cfg: AttrDict) -> None:
    data_loader_cfg = getattr(cfg, "data_loader", AttrDict({}))

    folder = data_loader_cfg.get("folder")
    if not folder:
        default_folder = Path(cfg.data_path) / ".combined" / cfg.identifier
        folder = str(default_folder)
    data_loader_cfg.folder = folder

    auto_load = data_loader_cfg.get("auto_load", True)
    if isinstance(auto_load, str):
        auto_load = auto_load.lower() == "true"
    data_loader_cfg.auto_load = bool(auto_load)

    cfg.data_loader = data_loader_cfg


if config_file := os.getenv("EXAREME2_WORKER_CONFIG_FILE"):
    with open(config_file) as fp:
        config = AttrDict(envtoml.load(fp))
else:
    with open_text(worker, "config.toml") as fp:
        config = AttrDict(envtoml.load(fp))

_ensure_duckdb_config(config)
_ensure_data_loader_config(config)
