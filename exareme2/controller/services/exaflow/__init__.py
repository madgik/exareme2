from typing import Optional

from exareme2.controller import config as ctrl_config
from exareme2.controller.services import get_worker_landscape_aggregator
from exareme2.controller.services.exaflow.controller import Controller

_exaflow_ctrl: Optional[Controller] = None


def set_exaflow_controller(controller: Controller):
    global _exaflow_ctrl
    _exaflow_ctrl = controller


def get_exaflow_controller() -> Controller:
    global _exaflow_ctrl
    if _exaflow_ctrl is None:
        worker_landscape = get_worker_landscape_aggregator()
        timeout = getattr(ctrl_config, "task_timeout", 0)
        _exaflow_ctrl = Controller(
            worker_landscape_aggregator=worker_landscape,
            task_timeout=timeout,
        )
    return _exaflow_ctrl
