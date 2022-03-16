from typing import Dict

from pydantic import BaseModel

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
    values: Dict[str, CommonDataElement]

    def __eq__(self, other):
        """
        We are overriding the equals function to check that the two cdes have identical fields except one edge case.
        The edge case is that the two comparing cdes can only contain a difference in the field of enumerations in
        the cde with code 'dataset' and still be considered compatible.
        """
        if set(self.values.keys()) != set(other.values.keys()):
            return False
        for cde_code in self.values.keys():
            cde1 = self.values[cde_code]
            cde2 = other.values[cde_code]
            if not cde1 == cde2 and not _are_equal_dataset_cdes(cde1, cde2):
                return False
        return True
