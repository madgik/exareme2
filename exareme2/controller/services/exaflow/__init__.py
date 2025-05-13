from typing import Optional

from exareme2.controller.services.exaflow.controller import Controller

_controller: Optional[Controller] = None


def set_controller(controller: Controller):
    global _controller
    _controller = controller


def get_controller() -> Controller:
    global _controller
    if not _controller:
        raise ValueError("Controller has not been initialized.")
    return _controller
