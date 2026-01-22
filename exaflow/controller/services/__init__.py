from typing import Optional

from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    WorkerLandscapeAggregator,
)

_worker_landscape_aggregator: Optional[WorkerLandscapeAggregator] = None


def set_worker_landscape_aggregator(
    worker_landscape_aggregator: WorkerLandscapeAggregator,
):
    global _worker_landscape_aggregator
    _worker_landscape_aggregator = worker_landscape_aggregator


def get_worker_landscape_aggregator() -> WorkerLandscapeAggregator:
    global _worker_landscape_aggregator
    if not _worker_landscape_aggregator:
        raise ValueError("WorkerLandscapeAggregator has not been initialized.")
    return _worker_landscape_aggregator
