from typing import Optional

from exareme2.controller.services.exareme2.cleaner import Cleaner
from exareme2.controller.services.exareme2.controller import Exareme2Controller

_exareme2_cleaner: Optional[Cleaner] = None
_exareme2_controller: Optional[Exareme2Controller] = None


def set_exareme2_cleaner(cleaner: Cleaner):
    global _exareme2_cleaner
    _exareme2_cleaner = cleaner


def get_exareme2_cleaner() -> Cleaner:
    global _exareme2_cleaner
    if not _exareme2_cleaner:
        raise ValueError("Exareme2 cleaner has not been initialized.")
    return _exareme2_cleaner


def set_exareme2_controller(controller: Exareme2Controller):
    global _exareme2_controller
    _exareme2_controller = controller


def get_controller() -> Exareme2Controller:
    global _exareme2_controller
    if not _exareme2_controller:
        raise ValueError("Exareme2 controller has not been initialized.")
    return _exareme2_controller
