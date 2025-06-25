from typing import Optional

from exareme2.controller.services.flower.controller import FlowerController
from exareme2.controller.services.flower.flower_io_registry import FlowerIORegistry

_flower_execution_info: Optional[FlowerIORegistry] = None
_controller: Optional[FlowerController] = None


def set_controller(controller: FlowerController):
    global _controller
    _controller = controller


def get_controller() -> FlowerController:
    global _controller
    if not _controller:
        raise ValueError("Controller has not been initialized.")
    return _controller


def set_flower_execution_info(flower_execution_info: FlowerIORegistry):
    global _flower_execution_info
    _flower_execution_info = flower_execution_info


def get_flower_execution_info() -> FlowerIORegistry:
    global _flower_execution_info
    return _flower_execution_info
