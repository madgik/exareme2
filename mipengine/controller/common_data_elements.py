from typing import Dict

from pydantic import BaseModel

from mipengine.node_exceptions import IncompatibleCDEs
from mipengine.node_tasks_DTOs import CommonDataElement


def _are_equal_dataset_cdes(cde1: CommonDataElement, cde2: CommonDataElement) -> bool:
    if cde1.code != "dataset" or cde2.code != "dataset":
        return False

    if (
        cde1.label != cde2.label
        or cde1.sql_type != cde2.sql_type
        or cde1.is_categorical != cde2.is_categorical
        or cde1.max != cde2.max
        or cde1.min != cde2.min
    ):
        return False

    return True


class CommonDataElements(BaseModel):
    cdes: Dict[str, CommonDataElement]

    def __eq__(self, other):
        if set(self.cdes.keys()) != set(other.cdes.keys()):
            raise IncompatibleCDEs(self.cdes, other.cdes)
        for cde_code in self.cdes.keys():
            cde1 = self.cdes[cde_code]
            cde2 = other.cdes[cde_code]
            if cde1 != cde2 and not _are_equal_dataset_cdes(cde1, cde2):
                raise IncompatibleCDEs(self.cdes, other.cdes)
        return True
