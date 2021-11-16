from string import Template
from typing import List
from typing import Optional


class UDFOutputTable:
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


class UDFExecutionQueries:
    output_tables: List[UDFOutputTable]
    udf_definition_query: Optional[Template]
    udf_select_query: Template

    def __init__(
        self,
        output_tables,
        udf_select_query,
        udf_definition_query=None,
    ):
        self.output_tables = output_tables
        self.udf_definition_query = udf_definition_query
        self.udf_select_query = udf_select_query
