from typing import Optional

from exareme2.controller import config as ctrl_config
from exareme2.controller.services import get_worker_landscape_aggregator
from exareme2.controller.services.exaflow.controller import ExaflowController

_exaflow_ctrl: Optional[ExaflowController] = None


def set_exaflow_controller(controller: ExaflowController):
    global _exaflow_ctrl
    _exaflow_ctrl = controller


def get_exaflow_controller() -> ExaflowController:
    global _exaflow_ctrl
    if _exaflow_ctrl is None:
        worker_landscape = get_worker_landscape_aggregator()
        timeout = getattr(ctrl_config, "task_timeout", 0)
        _exaflow_ctrl = ExaflowController(
            worker_landscape_aggregator=worker_landscape,
            task_timeout=timeout,
        )
    return _exaflow_ctrl
