from typing import Optional

from exareme2.controller import config as ctrl_config
from exareme2.controller.services import get_worker_landscape_aggregator
from exareme2.controller.services.exaflow.aggregation_server_exaflow_controller import (
    AggregationServerExaflowController,
)
from exareme2.controller.services.exaflow.exaflow_controller import ExaflowController

# Singletons for each controller flavor
_exaflow_ctrl: Optional[ExaflowController] = None
_exaflow_agg_ctrl: Optional[AggregationServerExaflowController] = None


def set_exaflow_controller(controller: ExaflowController):
    """Override the default ExaFlow (non-aggregation_server) controller."""
    global _exaflow_ctrl
    _exaflow_ctrl = controller


def set_aggregation_server_exaflow_controller(
    controller: AggregationServerExaflowController,
):
    """Override the default ExaFlow aggregation_server controller."""
    global _exaflow_agg_ctrl
    _exaflow_agg_ctrl = controller


def get_exaflow_controller() -> ExaflowController:
    """Get or initialize the standard ExaFlow controller."""
    global _exaflow_ctrl
    if _exaflow_ctrl is None:
        worker_landscape = get_worker_landscape_aggregator()
        timeout = getattr(ctrl_config, "task_timeout", 0)
        _exaflow_ctrl = ExaflowController(
            worker_landscape_aggregator=worker_landscape,
            task_timeout=timeout,
        )
    return _exaflow_ctrl


def get_aggregation_server_exaflow_controller() -> AggregationServerExaflowController:
    """Get or initialize the ExaFlow controller with aggregation server."""
    global _exaflow_agg_ctrl
    if _exaflow_agg_ctrl is None:
        worker_landscape = get_worker_landscape_aggregator()
        timeout = getattr(ctrl_config, "task_timeout", 0)
        _exaflow_agg_ctrl = AggregationServerExaflowController(
            worker_landscape_aggregator=worker_landscape,
            task_timeout=timeout,
        )
    return _exaflow_agg_ctrl
