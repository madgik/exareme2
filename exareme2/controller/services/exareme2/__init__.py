from typing import Optional

from exareme2.controller.services.exareme2.cleaner import Cleaner
from exareme2.controller.services.exareme2.controller import Exareme2Controller

_cleaner: Optional[Cleaner] = None
_controller: Optional[Exareme2Controller] = None


def set_cleaner(cleaner: Cleaner):
    global _cleaner
    _cleaner = cleaner


def get_cleaner() -> Cleaner:
    global _cleaner
    if not _cleaner:
        raise ValueError("Cleaner has not been initialized.")
    return _cleaner


def set_controller(controller: Exareme2Controller):
    global _controller
    _controller = controller


def get_controller() -> Exareme2Controller:
    global _controller
    if not _controller:
        raise ValueError("Controller has not been initialized.")
    return _controller
