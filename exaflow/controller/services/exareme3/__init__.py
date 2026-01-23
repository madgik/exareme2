from typing import Optional

from exaflow.controller import config as ctrl_config
from exaflow.controller.services import get_worker_landscape_aggregator
from exaflow.controller.services.exareme3.controller import Exareme3Controller

_exareme3_controller: Optional[Exareme3Controller] = None


def set_exareme3_controller(controller: Exareme3Controller):
    global _exareme3_controller
    _exareme3_controller = controller


def get_exareme3_controller() -> Exareme3Controller:
    global _exareme3_controller
    if _exareme3_controller is None:
        worker_landscape = get_worker_landscape_aggregator()
        timeout = getattr(ctrl_config, "task_timeout", 0)
        _exareme3_controller = Exareme3Controller(
            worker_landscape_aggregator=worker_landscape,
            task_timeout=timeout,
        )
    return _exareme3_controller
