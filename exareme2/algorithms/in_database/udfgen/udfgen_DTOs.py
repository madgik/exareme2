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
    share: bool = False


class UDFGenSMPCResult(UDFGenResult):
    template: UDFGenTableResult
    sum_op_values: Optional[UDFGenTableResult] = None
    min_op_values: Optional[UDFGenTableResult] = None
    max_op_values: Optional[UDFGenTableResult] = None
