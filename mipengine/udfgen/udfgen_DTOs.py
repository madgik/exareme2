from string import Template
from typing import List
from typing import Optional

from mipengine.node_tasks_DTOs import TableInfo


class UDFGenResult:
    pass


class TableUDFGenResult(UDFGenResult):
    tablename_placeholder: str
    drop_query: Template
    create_query: Template

    def __init__(
        self,
        tablename_placeholder,
        drop_query,
        create_query,
    ):
        self.tablename_placeholder = tablename_placeholder
        self.drop_query = drop_query
        self.create_query = create_query

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
    add_op_values: TableUDFGenResult
    min_op_values: TableUDFGenResult
    max_op_values: TableUDFGenResult
    union_op_values: TableUDFGenResult

    def __init__(
        self,
        template,
        add_op_values=None,
        min_op_values=None,
        max_op_values=None,
        union_op_values=None,
    ):
        self.template = template
        self.add_op_values = add_op_values
        self.min_op_values = min_op_values
        self.max_op_values = max_op_values
        self.union_op_values = union_op_values

    def __eq__(self, other):
        if self.template != other.template:
            return False
        if self.add_op_values != other.add_op_values:
            return False
        if self.min_op_values != other.min_op_values:
            return False
        if self.max_op_values != other.max_op_values:
            return False
        if self.union_op_values != other.union_op_values:
            return False
        return True

    def __repr__(self):
        return (
            f"SMPCUDFGenResult("
            f"template={self.template}, "
            f"add_op_values={self.add_op_values}, "
            f"min_op_values={self.min_op_values}, "
            f"max_op_values={self.max_op_values}, "
            f"union_op_values={self.union_op_values}"
            f")"
        )


class UDFGenExecutionQueries:
    udf_results: List[UDFGenResult]
    udf_definition_query: Optional[Template]
    udf_select_query: Template

    def __init__(
        self,
        udf_results,
        udf_select_query,
        udf_definition_query=None,
    ):
        self.udf_results = udf_results
        self.udf_definition_query = udf_definition_query
        self.udf_select_query = udf_select_query

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


class SMPCTablesInfo:
    template: TableInfo
    add_op_values: TableInfo
    min_op_values: TableInfo
    max_op_values: TableInfo
    union_op_values: TableInfo

    def __init__(
        self,
        template,
        add_op_values=None,
        min_op_values=None,
        max_op_values=None,
        union_op_values=None,
    ):
        self.template = template
        self.add_op_values = add_op_values
        self.min_op_values = min_op_values
        self.max_op_values = max_op_values
        self.union_op_values = union_op_values

    def __repr__(self):
        return (
            f"SMPCUDFInput("
            f"template={self.template}, "
            f"add_op_values={self.add_op_values}, "
            f"min_op_values={self.min_op_values}, "
            f"max_op_values={self.max_op_values}, "
            f"union_op_values={self.union_op_values}"
            f")"
        )
