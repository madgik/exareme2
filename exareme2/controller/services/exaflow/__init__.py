from typing import Optional

from exareme2.controller import config as ctrl_config
from exareme2.controller.services import get_worker_landscape_aggregator
from exareme2.controller.services.exaflow.controller import ExaflowController

_exaflow_controller: Optional[ExaflowController] = None


def set_exaflow_controller(controller: ExaflowController):
    global _exaflow_controller
    _exaflow_controller = controller


def get_exaflow_controller() -> ExaflowController:
    global _exaflow_controller
    if _exaflow_controller is None:
        worker_landscape = get_worker_landscape_aggregator()
        timeout = getattr(ctrl_config, "task_timeout", 0)
        _exaflow_controller = ExaflowController(
            worker_landscape_aggregator=worker_landscape,
            task_timeout=timeout,
        )
    return _exaflow_controller
