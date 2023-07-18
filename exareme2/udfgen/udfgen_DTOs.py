from typing import List
from typing import Optional
from typing import Tuple

from pydantic import BaseModel

from exareme2 import DType


class UDFGenResult(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False


class UDFGenTableResult(UDFGenResult):
    table_name: str
    table_schema: List[Tuple[str, DType]]
    create_query: str


class UDFGenSMPCResult(UDFGenResult):
    template: UDFGenTableResult
    sum_op_values: Optional[UDFGenTableResult] = None
    min_op_values: Optional[UDFGenTableResult] = None
    max_op_values: Optional[UDFGenTableResult] = None

    @property
    def create_query(self):
        queries = [self.template.create_query]
        if self.sum_op_values is not None:
            queries.append(self.sum_op_values.create_query)
        if self.min_op_values is not None:
            queries.append(self.min_op_values.create_query)
        if self.max_op_values is not None:
            queries.append(self.max_op_values.create_query)
        return "".join(queries)
