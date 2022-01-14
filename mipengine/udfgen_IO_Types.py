from abc import ABC
from abc import abstractmethod
from string import Template
from typing import List
from typing import Optional

from typing import Tuple

from typing import Dict

from mipengine.node.monetdb_interface.common_actions import create_table_name
from mipengine.node_tasks_DTOs import NodeSMPCDTO
from mipengine.node_tasks_DTOs import NodeSMPCValueDTO
from mipengine.node_tasks_DTOs import TableInfo

from mipengine.node import config as node_config
from mipengine.node_tasks_DTOs import TableType
from mipengine.node_tasks_DTOs import NodeTableDTO
from mipengine.node_tasks_DTOs import NodeUDFDTO
from mipengine.node_tasks_DTOs import UDFResults


class UDFGenResult(ABC):
    @abstractmethod
    def convert_to_udf_result_and_mapping(
        self,
        context_id: str,
        command_id: str,
        command_subid: int,
    ) -> Tuple[NodeUDFDTO, Dict[str, str]]:
        raise NotImplementedError


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

    def convert_to_udf_result_and_mapping(
        self,
        context_id: str,
        command_id: str,
        command_subid: int,
    ) -> Tuple[NodeTableDTO, Dict[str, str]]:
        table_name_ = create_table_name(
            table_type=TableType.NORMAL,
            node_id=node_config.identifier,
            context_id=context_id,
            command_id=command_id,
            command_subid=str(command_subid),
        )
        table_name_tmpl_mapping = {self.tablename_placeholder: table_name_}
        return NodeTableDTO(value=table_name_), table_name_tmpl_mapping


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

    def convert_to_udf_result_and_mapping(
        self,
        context_id: str,
        command_id: str,
        command_subid: int,
    ) -> Tuple[NodeSMPCDTO, Dict[str, str]]:

        (
            template_udf_result,
            table_names_tmpl_mapping,
        ) = self.template.convert_to_udf_result_and_mapping(
            context_id, command_id, command_subid
        )

        if self.add_op_values:
            (
                add_op_udf_result,
                mapping,
            ) = self.add_op_values.convert_to_udf_result_and_mapping(
                context_id, command_id, command_subid + 1
            )
            table_names_tmpl_mapping.update(mapping)
        else:
            add_op_udf_result = None

        if self.min_op_values:
            (
                min_op_udf_result,
                mapping,
            ) = self.min_op_values.convert_to_udf_result_and_mapping(
                context_id, command_id, command_subid + 2
            )
            table_names_tmpl_mapping.update(mapping)
        else:
            min_op_udf_result = None

        if self.max_op_values:
            (
                max_op_udf_result,
                mapping,
            ) = self.max_op_values.convert_to_udf_result_and_mapping(
                context_id, command_id, command_subid + 3
            )
            table_names_tmpl_mapping.update(mapping)
        else:
            max_op_udf_result = None

        if self.union_op_values:
            (
                union_op_udf_result,
                mapping,
            ) = self.union_op_values.convert_to_udf_result_and_mapping(
                context_id, command_id, command_subid + 4
            )
            table_names_tmpl_mapping.update(mapping)
        else:
            union_op_udf_result = None

        result = NodeSMPCDTO(
            value=NodeSMPCValueDTO(
                template=template_udf_result,
                add_op_values=add_op_udf_result,
                min_op_values=min_op_udf_result,
                max_op_values=max_op_udf_result,
                union_op_values=union_op_udf_result,
            )
        )

        return result, table_names_tmpl_mapping


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

    def convert_to_udf_result_and_mapping(
        self,
        context_id: str,
        command_id: str,
    ) -> Tuple[UDFResults, Dict[str, str]]:
        """
        Iterates through all the udf generator results, in order to create
        a table for each one.

        UDFResults are returned together with a mapping of
        template -> tablename, so it can be used in the udf's declaration to
        replace the templates with the actual table names.

        Returns
        -------
        a UDFResults object containing all the results
        a dictionary of template (placeholder) tablename to the actual table name.
        """

        results = []
        table_names_tmpl_mapping = {}
        command_subid = 0
        for udf_result in self.udf_results:
            table_name, mapping = udf_result.convert_to_udf_result_and_mapping(
                context_id,
                command_id,
                command_subid,
            )
            table_names_tmpl_mapping.update(mapping)
            results.append(table_name)

            # Needs to be incremented by 10 because a udf_result could
            # contain more than one tables. (SMPC for example)
            command_subid += 10

        udf_results = UDFResults(results=results)
        return udf_results, table_names_tmpl_mapping


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
