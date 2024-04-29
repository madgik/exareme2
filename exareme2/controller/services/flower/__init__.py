from typing import Optional

from exareme2.controller.services.flower.controller import Controller
from exareme2.controller.services.flower.flower_execution_info import (
    FlowerExecutionInfo,
)

_flower_experiment_watcher: Optional[FlowerExecutionInfo] = None
_controller: Optional[Controller] = None


def set_controller(controller: Controller):
    global _controller
    _controller = controller


def get_controller() -> Controller:
    global _controller
    if not _controller:
        raise ValueError("Controller has not been initialized.")
    return _controller


def set_flower_experiment_watcher(flower_experiment_watcher: FlowerExecutionInfo):
    global _flower_experiment_watcher
    _flower_experiment_watcher = flower_experiment_watcher


def get_flower_experiment_watcher() -> FlowerExecutionInfo:
    global _flower_experiment_watcher
    return _flower_experiment_watcher
