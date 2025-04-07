from enum import Enum


class AGG(Enum):
    SUM = "SUM"
    MIN = "MIN"
    MAX = "MAX"

    def __str__(self):
        return self.name
