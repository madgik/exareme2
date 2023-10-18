from typing import Optional

from exareme2.controller.services.node_landscape_aggregator import (
    NodeLandscapeAggregator,
)

_node_landscape_aggregator: Optional[NodeLandscapeAggregator] = None


def set_node_landscape_aggregator(node_landscape_aggregator: NodeLandscapeAggregator):
    global _node_landscape_aggregator
    _node_landscape_aggregator = node_landscape_aggregator


def get_node_landscape_aggregator() -> NodeLandscapeAggregator:
    global _node_landscape_aggregator
    if not _node_landscape_aggregator:
        raise ValueError("NodeLandscapeAggregator has not been initialized.")
    return _node_landscape_aggregator
