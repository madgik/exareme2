from typing import Optional

from exareme2.controller.services.flower.controller import FlowerController

_flower_controller: Optional[FlowerController] = None


def set_flower_controller(flower_controller: FlowerController):
    global _flower_controller
    _flower_controller = flower_controller


def get_flower_controller() -> FlowerController:
    global _flower_controller
    if not _flower_controller:
        raise ValueError("Controller has not been initialized.")
    return _flower_controller
