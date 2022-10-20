from string import Template
from typing import List
from typing import Optional
from typing import Tuple

from pydantic import BaseModel

from mipengine import DType


class UDFGenBaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False


class UDFGenResult(UDFGenBaseModel):
    pass


class UDFGenTableResult(UDFGenResult):
    tablename_placeholder: str
    table_schema: List[Tuple[str, DType]]
    drop_query: Template
    create_query: Template

    def __eq__(self, other):
        if self.tablename_placeholder != other.tablename_placeholder:
            return False
        if self.table_schema != other.table_schema:
            return False
        if self.drop_query.template != other.drop_query.template:
            return False
        if self.create_query.template != other.create_query.template:
            return False
        return True

    def __repr__(self):
        return (
            f"UDFGenTableResult("
            f"{self.tablename_placeholder=}, "
            f"{self.table_schema=}, "
            f"{self.drop_query.template=}, "
            f"{self.create_query.template=}"
            f")"
        )


class UDFGenSMPCResult(UDFGenResult):
    template: UDFGenTableResult
    sum_op_values: Optional[UDFGenTableResult] = None
    min_op_values: Optional[UDFGenTableResult] = None
    max_op_values: Optional[UDFGenTableResult] = None

    def __eq__(self, other):
        if self.template != other.template:
            return False
        if self.sum_op_values != other.sum_op_values:
            return False
        if self.min_op_values != other.min_op_values:
            return False
        if self.max_op_values != other.max_op_values:
            return False
        return True

    def __repr__(self):
        return (
            f"UDFGenSMPCResult("
            f"template={self.template}, "
            f"sum_op_values={self.sum_op_values}, "
            f"min_op_values={self.min_op_values}, "
            f"max_op_values={self.max_op_values}, "
            f")"
        )


class UDFGenExecutionQueries(UDFGenBaseModel):
    udf_results: List[UDFGenResult]
    udf_definition_query: Optional[Template] = None
    udf_select_query: Template

    def __repr__(self):
        udf_definition_query_str = "None"
        if self.udf_definition_query:
            udf_definition_query_str = self.udf_definition_query.template
        return (
            f"UDFExecutionQueries("
            f"udf_results={self.udf_results}, "
            f"udf_definition_query='{udf_definition_query_str}', "
            f"udf_select_query='{self.udf_select_query.template}'"
            f")"
        )
