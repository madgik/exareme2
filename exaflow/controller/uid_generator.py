from random import randint
from threading import Lock

from exaflow.utils import Singleton


class UIDGenerator(metaclass=Singleton):
    def __init__(self):
        self._uid_generation_lock = Lock()
        self._uid_max_num_chars = 9

        # The first number should be random to avoid conflicts in dev deployments
        self._current_number = randint(0, (10**self._uid_max_num_chars) - 1)

    def _increment_current_number(self):
        if self._current_number >= 10**self._uid_max_num_chars:
            self._current_number = 0
        else:
            self._current_number += 1

    def get_a_uid(self):
        with self._uid_generation_lock:
            self._increment_current_number()
            return str(self._current_number).rjust(self._uid_max_num_chars, "0")
