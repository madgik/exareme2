from typing import Optional

from exareme2.controller.services.exareme2.cleaner import Cleaner
from exareme2.controller.services.exareme2.controller import Exareme2Controller

_exareme2_cleaner: Optional[Cleaner] = None
_flower_controller: Optional[Exareme2Controller] = None


def set_cleaner(cleaner: Cleaner):
    global _cleaner
    _cleaner = cleaner


def get_exareme2_cleaner() -> Cleaner:
    global _exareme2_cleaner
    if not _exareme2_cleaner:
        raise ValueError("Exareme2 cleaner has not been initialized.")
    return _exareme2_cleaner


def set_controller(controller: Exareme2Controller):
    global _flower_controller
    _controller = controller


def get_controller() -> Exareme2Controller:
    global _flower_controller
    if not _flower_controller:
        raise ValueError("Controller has not been initialized.")
    return _flower_controller
