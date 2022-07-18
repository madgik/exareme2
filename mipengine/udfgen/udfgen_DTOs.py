from string import Template
from typing import List
from typing import Optional

from pydantic import BaseModel

from mipengine.node_tasks_DTOs import TableInfo


class UDFGenBaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False


class UDFGenResult(UDFGenBaseModel):
    pass


class TableUDFGenResult(UDFGenResult):
    tablename_placeholder: str
    drop_query: Template
    create_query: Template

    def __eq__(self, other):
        if self.tablename_placeholder != other.tablename_placeholder:
            return False
        if self.drop_query.template != other.drop_query.template:
            return False
        if self.create_query.template != other.create_query.template:
            return False
        return True

    def __repr__(self):
        return (
            f"TableUDFGenResult("
            f"tablename_placeholder='{self.tablename_placeholder}', "
            f"drop_query='{self.drop_query.template}', "
            f"create_query='{self.create_query.template}'"
            f")"
        )


class SMPCUDFGenResult(UDFGenResult):
    template: TableUDFGenResult
    sum_op_values: Optional[TableUDFGenResult] = None
    min_op_values: Optional[TableUDFGenResult] = None
    max_op_values: Optional[TableUDFGenResult] = None

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
            f"SMPCUDFGenResult("
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


class SMPCTablesInfo(UDFGenBaseModel):
    template: TableInfo
    sum_op_values: Optional[TableInfo] = None
    min_op_values: Optional[TableInfo] = None
    max_op_values: Optional[TableInfo] = None

    def __repr__(self):
        return (
            f"SMPCUDFInput("
            f"template={self.template}, "
            f"sum_op_values={self.sum_op_values}, "
            f"min_op_values={self.min_op_values}, "
            f"max_op_values={self.max_op_values}, "
            f")"
        )
