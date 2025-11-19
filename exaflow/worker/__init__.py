import os
from importlib.resources import open_text
from pathlib import Path

import envtoml

from exaflow import AttrDict
from exaflow import worker


def _ensure_duckdb_config(cfg: AttrDict) -> None:
    duckdb_cfg = getattr(cfg, "duckdb", AttrDict({}))
    duckdb_path = duckdb_cfg.get("path")
    if not duckdb_path:
        default_path = Path(cfg.data_path) / f"data_models.duckdb"
        duckdb_path = str(default_path)
    duckdb_cfg.path = duckdb_path
    cfg.duckdb = duckdb_cfg


def _ensure_grpc_config(cfg: AttrDict) -> None:
    grpc_cfg = getattr(cfg, "grpc", AttrDict({}))

    default_ip = getattr(getattr(cfg, "grpc", AttrDict({})), "ip", "0.0.0.0")
    default_port = getattr(getattr(cfg, "grpc", AttrDict({})), "port", 50051)

    grpc_cfg.ip = grpc_cfg.get("ip", default_ip)
    grpc_cfg.port = int(grpc_cfg.get("port", default_port))

    cfg.grpc = grpc_cfg


def _ensure_worker_tasks_config(cfg: AttrDict) -> None:
    worker_tasks_cfg = getattr(cfg, "worker_tasks", AttrDict({}))

    tasks_timeout = worker_tasks_cfg.get("tasks_timeout", 120)

    worker_tasks_cfg.tasks_timeout = int(tasks_timeout)

    cfg.worker_tasks = worker_tasks_cfg


if config_file := os.getenv("EXAFLOW_WORKER_CONFIG_FILE"):
    with open(config_file) as fp:
        config = AttrDict(envtoml.load(fp))
else:
    with open_text(worker, "config.toml") as fp:
        config = AttrDict(envtoml.load(fp))

_ensure_duckdb_config(config)
_ensure_grpc_config(config)
_ensure_worker_tasks_config(config)
