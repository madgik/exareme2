import ast
from abc import ABC
from abc import abstractmethod
from numbers import Number
from textwrap import indent
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import astor

from mipengine import DType
from mipengine.udfgen.helpers import get_func_body_from_ast
from mipengine.udfgen.helpers import get_items_of_type
from mipengine.udfgen.helpers import get_return_names_from_body
from mipengine.udfgen.helpers import iotype_to_sql_schema
from mipengine.udfgen.helpers import is_any_element_of_type
from mipengine.udfgen.helpers import make_unique_func_name
from mipengine.udfgen.helpers import parse_func
from mipengine.udfgen.helpers import recursive_repr
from mipengine.udfgen.helpers import remove_empty_lines
from mipengine.udfgen.iotypes import InputType
from mipengine.udfgen.iotypes import LiteralArg
from mipengine.udfgen.iotypes import LiteralType
from mipengine.udfgen.iotypes import LoopbackOutputType
from mipengine.udfgen.iotypes import OutputType
from mipengine.udfgen.iotypes import ParametrizedType
from mipengine.udfgen.iotypes import PlaceholderArg
from mipengine.udfgen.iotypes import StateType
from mipengine.udfgen.iotypes import TableArg
from mipengine.udfgen.iotypes import TableType
from mipengine.udfgen.iotypes import TransferTypeBase
from mipengine.udfgen.iotypes import UDFLoggerArg
from mipengine.udfgen.iotypes import UDFLoggerType

CREATE_OR_REPLACE_FUNCTION = "CREATE OR REPLACE FUNCTION"
RETURNS = "RETURNS"
LANGUAGE_PYTHON = "LANGUAGE PYTHON"
BEGIN = "{"
END = "}"
SELECT = "SELECT"
FROM = "FROM"
WHERE = "WHERE"
GROUP_BY = "GROUP BY"
ORDER_BY = "ORDER BY"
AND = "AND"

SEP = ","
LN = "\n"
SEPLN = SEP + LN
ANDLN = " " + AND + LN
SPC4 = " " * 4


class Signature(NamedTuple):
    parameters: Dict[str, InputType]
    return_annotations: List[OutputType]


class FunctionParts(NamedTuple):
    """A function's parts, used in various stages of the udf definition/query
    generation."""

    qualname: str
    body_statements: list
    return_names: List[str]
    table_input_types: Dict[str, TableType]
    literal_input_types: Dict[str, LiteralType]
    logger_param_name: Optional[str]
    output_types: List[OutputType]
    sig: Signature


def breakup_function(func, funcsig) -> FunctionParts:
    """Breaks up a function into smaller parts, which will be used during
    the udf translation process."""
    qualname = make_unique_func_name(func)
    tree = parse_func(func)
    body_statements = get_func_body_from_ast(tree)
    return_names = get_return_names_from_body(body_statements)
    table_input_types = get_items_of_type(TableType, funcsig.parameters)
    literal_input_types = get_items_of_type(LiteralType, funcsig.parameters)

    logger_input_types = get_items_of_type(UDFLoggerType, funcsig.parameters)
    assert len(logger_input_types) < 2  # Only one logger is allowed
    logger_param_name = next(iter(logger_input_types.keys()), None)

    return FunctionParts(
        qualname=qualname,
        body_statements=body_statements,
        return_names=return_names,
        table_input_types=table_input_types,
        literal_input_types=literal_input_types,
        logger_param_name=logger_param_name,
        output_types=funcsig.return_annotations,
        sig=funcsig,
    )


class ASTNode(ABC):
    @abstractmethod
    def compile(self) -> str:
        raise NotImplementedError

    __repr__ = recursive_repr

    def __str__(self):
        return self.compile()  # pragma: no cover


class UDFParameter(ASTNode):
    def __init__(self, param_arg, name="") -> None:
        self.name = name
        self.input_type = param_arg.type

    def compile(self) -> str:
        return iotype_to_sql_schema(iotype=self.input_type, name_prefix=self.name)


class UDFReturnType(ASTNode):
    def __init__(self, return_type) -> None:
        self.output_type = return_type

    def compile(self) -> str:
        return self.output_type.get_return_type_template()


class UDFSignature(ASTNode):
    def __init__(
        self,
        udfname: str,
        table_args: Dict[str, TableArg],
        return_type: OutputType,
    ):
        self.udfname = udfname
        self.parameter_types = [
            UDFParameter(arg, name)
            for name, arg in table_args.items()
            if isinstance(arg.type, ParametrizedType)
        ]
        self.return_type = UDFReturnType(return_type)

    def compile(self) -> str:
        return LN.join(
            [
                self._format_prototype(),
                RETURNS,
                self.return_type.compile(),
            ]
        )

    def _format_prototype(self):
        return f"{self.udfname}({self._format_parameters()})"

    def _format_parameters(self):
        return SEP.join(param.compile() for param in self.parameter_types)


class UDFHeader(ASTNode):
    def __init__(
        self,
        udfname: str,
        table_args: Dict[str, TableArg],
        return_type: OutputType,
    ):
        self.signature = UDFSignature(udfname, table_args, return_type)

    def compile(self) -> str:
        return LN.join(
            [
                CREATE_OR_REPLACE_FUNCTION,
                self.signature.compile(),
                LANGUAGE_PYTHON,
            ]
        )


class Imports(ASTNode):
    # TODO imports should be dynamic
    def __init__(self, import_pickle, import_json):
        self._import_lines = ["import pandas as pd", "import udfio"]
        if import_pickle:
            pickle_import = "import pickle"
            self._import_lines.append(pickle_import)
        if import_json:
            json_import = "import json"
            self._import_lines.append(json_import)

    def compile(self) -> str:
        return LN.join(self._import_lines)


class TableBuild(ASTNode):
    def __init__(self, arg_name, arg, template):
        self.arg_name = arg_name
        self.arg = arg
        self.template = template

    def compile(self) -> str:
        colnames = self.arg.type.column_names()
        colnames_w_prefix = self.arg.type.column_names(prefix=self.arg_name)
        return self.template.format(
            varname=self.arg_name,
            colnames=colnames,
            colnames_w_prefix=colnames_w_prefix,
            table_name=self.arg.table_name,
        )


class TableBuilds(ASTNode):
    def __init__(self, table_args: Dict[str, TableArg]):
        self.table_builds = [
            TableBuild(arg_name, arg, template=arg.type.get_build_template())
            for arg_name, arg in table_args.items()
        ]

    def compile(self) -> str:
        return LN.join([tb.compile() for tb in self.table_builds])


class UDFReturnStatement(ASTNode):
    def __init__(self, return_name, return_type):
        self.return_name = return_name
        self.template = return_type.get_main_return_stmt_template()

    def compile(self) -> str:
        return self.template.format(return_name=self.return_name)


class UDFLoopbackReturnStatements(ASTNode):
    def __init__(self, sec_return_names, sec_return_types, sec_output_table_names):
        self.sec_return_names = sec_return_names
        assert len(sec_output_table_names) == len(sec_return_types)
        self.templates = [
            table.get_secondary_return_stmt_template(name)
            for name, table in zip(sec_output_table_names, sec_return_types)
        ]

    def compile(self) -> str:
        return LN.join(
            [
                template.format(return_name=return_name)
                for template, return_name in zip(self.templates, self.sec_return_names)
            ]
        )


def get_name_loopback_table_pairs(
    output_types: List[LoopbackOutputType],
) -> List[Tuple[str, LoopbackOutputType]]:
    """
    Receives a list of LoopbackOutputTypes and returns a list of pairs
    of table name placeholders and LoopbackOutputTypes.
    """
    return [
        (f"loopback_table_name_{pos}", output_type)
        for pos, output_type in enumerate(output_types)
    ]


class LiteralAssignments(ASTNode):
    def __init__(self, literals: Dict[str, LiteralArg]):
        self.literals = literals

    def compile(self) -> str:
        return LN.join(
            f"{name} = {repr(arg.value)}" for name, arg in self.literals.items()
        )


class LoggerAssignment(ASTNode):
    def __init__(self, logger: Optional[Tuple[str, UDFLoggerArg]]):
        self.logger = logger

    def compile(self) -> str:
        if not self.logger:
            return ""
        name, logger_arg = self.logger
        udf_name = logger_arg.udf_name
        request_id = logger_arg.request_id
        return f"{name} = udfio.get_logger('{udf_name}', '{request_id}')"


class PlaceholderAssignments(ASTNode):
    def __init__(self, placeholders):
        self.placeholders = placeholders

    def compile(self) -> str:
        return LN.join(
            f"{name} = {arg.name}" for name, arg in self.placeholders.items()
        )


class UDFBodyStatements(ASTNode):
    def __init__(self, statements):
        self.returnless_stmts = [
            astor.to_source(stmt)
            for stmt in statements
            if not isinstance(stmt, ast.Return)
        ]

    def compile(self) -> str:
        return LN.join(remove_empty_lines(self.returnless_stmts))


class UDFBody(ASTNode):
    def __init__(
        self,
        table_args: Dict[str, TableArg],
        literal_args: Dict[str, LiteralArg],
        logger_arg: Optional[Tuple[str, UDFLoggerArg]],
        placeholder_args: Dict[str, PlaceholderArg],
        statements: list,
        main_return_name: str,
        main_return_type: OutputType,
        sec_return_names: List[str],
        sec_return_types: List[OutputType],
        sec_output_table_names: List[str],
    ):
        all_types = (
            [arg.type for arg in table_args.values()]
            + [main_return_type]
            + sec_return_types
        )

        import_pickle = is_any_element_of_type(StateType, all_types)
        import_json = is_any_element_of_type(TransferTypeBase, all_types)

        self.statements = []

        # imports
        self.statements.append(
            Imports(
                import_pickle=import_pickle,
                import_json=import_json,
            )
        )

        # initial assignments
        self.statements.append(TableBuilds(table_args))
        self.statements.append(LiteralAssignments(literal_args))
        self.statements.append(LoggerAssignment(logger_arg))
        self.statements.append(PlaceholderAssignments(placeholder_args))

        # main body
        self.statements.append(UDFBodyStatements(statements))

        # return statements
        self.statements.append(
            UDFLoopbackReturnStatements(
                sec_return_names=sec_return_names,
                sec_return_types=sec_return_types,
                sec_output_table_names=sec_output_table_names,
            )
        )
        self.statements.append(UDFReturnStatement(main_return_name, main_return_type))

    def compile(self) -> str:
        return LN.join(remove_empty_lines([stmt.compile() for stmt in self.statements]))


class UDFDefinition(ASTNode):
    def __init__(self, header: UDFHeader, body: UDFBody):
        self.header = header
        self.body = body

    def compile(self) -> str:
        return LN.join(
            [
                self.header.compile(),
                BEGIN,
                indent(self.body.compile(), prefix=SPC4),
                END,
            ]
        )


class StarColumn(ASTNode):
    def compile(self, *_, **__):
        return "*"


class ConstColumn(ASTNode):
    def __init__(self, value, alias):
        self.name = str(value)
        self.alias = alias

    def compile(self, use_alias=False):
        return f"{self.name}" + (f' AS "{self.alias}"' if use_alias else "")


class Column(ASTNode):
    def __init__(
        self, name: str, table: Union["Table", str] = None, alias="", quote=True
    ):
        self.name = name
        self.table = table
        self.alias = alias
        self.quote = quote

    def compile(self, use_alias=True, use_prefix=True) -> str:
        prefix = self._get_prefix(use_prefix)
        postfix = f' AS "{self.alias}"' if self.alias and use_alias else ""
        if self.name == "*" or not self.quote:
            name = self.name
        else:
            name = f'"{self.name}"'
        return prefix + name + postfix

    def _get_prefix(self, use_prefix):
        # When prefix is used return a str of the form "table.column". The
        # prefix is the name of the table. If `table` is a Table, we get it
        # from `compile`. If `table` is a string it's just the table name.
        if self.table and use_prefix:
            if isinstance(self.table, Table):
                prefix = self.table.compile(use_alias=False) + "."
            elif isinstance(self.table, str):
                prefix = self.table + "."
            else:
                raise TypeError(f"table can be of type Table or str. Got {self.table}.")
        else:
            prefix = ""
        return prefix

    def __eq__(self, other):
        return ColumnEqualityClause(self, other)

    # XXX naive implementation ignoring operator precedence when more than 2 operands
    def __add__(self, other):
        if isinstance(other, Column):
            name1 = self.compile(use_alias=False)
            name2 = other.compile(use_alias=False)
            return Column(name=f"{name1} + {name2}", quote=False)
        if isinstance(other, Number):
            name1 = self.compile(use_alias=False)
            return Column(name=f"{name1} + {other}", quote=False)
        raise TypeError(f"unsupported operand types for +: Column and {type(other)}")

    def __radd__(self, other):
        if isinstance(other, Number):
            name1 = self.compile(use_alias=False)
            return Column(name=f"{other} + {name1}", quote=False)
        raise TypeError(f"unsupported operand types for +: {type(other)} and Column")

    def __sub__(self, other):
        if isinstance(other, Column):
            name1 = self.compile(use_alias=False)
            name2 = other.compile(use_alias=False)
            return Column(name=f"{name1} - {name2}", quote=False)
        if isinstance(other, Number):
            name1 = self.compile(use_alias=False)
            return Column(name=f"{name1} - {other}", quote=False)
        raise TypeError(f"unsupported operand types for -: Column and {type(other)}")

    def __rsub__(self, other):
        if isinstance(other, Number):
            name1 = self.compile(use_alias=False)
            return Column(name=f"{other} - {name1}", quote=False)
        raise TypeError(f"unsupported operand types for -: {type(other)} and Column")

    def __mul__(self, other):
        if isinstance(other, Column):
            name1 = self.compile(use_alias=False)
            name2 = other.compile(use_alias=False)
            return Column(name=f"{name1} * {name2}", quote=False)
        if isinstance(other, Number):
            name1 = self.compile(use_alias=False)
            return Column(name=f"{name1} * {other}", quote=False)
        raise TypeError(f"unsupported operand types for *: Column and {type(other)}")

    def __rmul__(self, other):
        if isinstance(other, Number):
            name1 = self.compile(use_alias=False)
            return Column(name=f"{other} * {name1}", quote=False)
        raise TypeError(f"unsupported operand types for *: {type(other)} and Column")

    def __truediv__(self, other):
        if isinstance(other, Column):
            name1 = self.compile(use_alias=False)
            name2 = other.compile(use_alias=False)
            return Column(name=f"{name1} / {name2}", quote=False)
        if isinstance(other, Number):
            name1 = self.compile(use_alias=False)
            return Column(name=f"{name1} / {other}", quote=False)
        raise TypeError(f"unsupported operand types for /: Column and {type(other)}")

    def __rtruediv__(self, other):
        if isinstance(other, Number):
            name1 = self.compile(use_alias=False)
            return Column(name=f"{other} / {name1}", quote=False)
        raise TypeError(f"unsupported operand types for /: {type(other)} and Column")


class Cast(ASTNode):
    def __init__(self, name, type_, alias):
        self.name = name
        self.type_ = type_
        self.alias = alias

    def compile(self, use_alias=False):
        result = f"CAST('{self.name}' AS {self.type_})"
        return result + (f" AS {self.alias}" if use_alias else "")


class ColumnEqualityClause(ASTNode):
    def __init__(self, column1: Union[Column, str], column2: Union[Column, str]):
        self.column1 = column1
        self.column2 = column2

    def compile(self) -> str:
        if isinstance(self.column1, Column):
            col1 = self.column1.compile(use_alias=False, use_prefix=True)
        elif isinstance(self.column1, str):
            col1 = f"'{self.column1}'"
        else:
            msg = f"Expected column of type Column or str. Got {self.column1}."
            raise TypeError(msg)
        if isinstance(self.column2, Column):
            col2 = self.column2.compile(use_alias=False, use_prefix=True)
        elif isinstance(self.column2, str):
            col2 = f"'{self.column2}'"
        else:
            msg = f"Expected column of type Column or str. Got {self.column2}."
            raise TypeError(msg)
        return col1 + "=" + col2


class ScalarFunction(ASTNode):
    def __init__(self, name: str, columns: List[Column], alias=""):
        self.name = name
        self.select_clause = ColumnsClauseParameters(columns, newline=False)
        self.alias = alias

    def compile(self, use_alias=False) -> str:
        postfix = f" AS {self.alias}" if self.alias and use_alias else ""
        return self.name + "(" + self.select_clause.compile() + ")" + postfix


class TableFunction(ASTNode):
    def __init__(self, name: str, subquery: "Select" = None, alias=""):
        self.name = name
        self.subquery = subquery
        self.alias = alias

    def compile(self, use_alias=False) -> str:
        postfix = f" AS {self.alias}" if self.alias and use_alias else ""
        if self.subquery:
            return LN.join(
                [
                    self.name + "((",
                    indent(self.subquery.compile(), prefix=SPC4),
                    "))" + postfix,
                ]
            )
        return self.name + "()" + postfix


class Table(ASTNode):
    def __init__(self, name: str, columns: List[Union[str, Column]], alias=""):
        self.name = name
        if columns and isinstance(columns[0], str):
            self.columns = {colname: Column(colname, self) for colname in columns}
        else:
            self.columns = columns
        self.c = self.columns
        self.alias = alias

    def compile(self, use_alias=False) -> str:
        if use_alias:
            postfix = f" AS {self.alias}" if self.alias else ""
            return self.name + postfix
        return self.alias or self.name


class Select(ASTNode):
    def __init__(
        self,
        columns: List[Union[Column, ScalarFunction]],
        from_: List[Union[Table, TableFunction]],
        where: List[ColumnEqualityClause] = None,
        groupby: List[Column] = None,
        orderby: List[Column] = None,
    ):
        self.select_clause = SelectClause(columns)
        self.from_clause = FromClause(from_)
        self.where_clause = WhereClause(where) if where else None
        self.groupby_clause = GroupbyClause(groupby) if groupby else None
        self.orderby_clause = OrderbyClause(orderby) if orderby else None

    def compile(self) -> str:
        lines = [
            self.select_clause.compile(),
            self.from_clause.compile(),
        ]
        if self.where_clause:
            lines += [
                self.where_clause.compile(),
            ]
        if self.groupby_clause:
            lines += [
                self.groupby_clause.compile(),
            ]
        if self.orderby_clause:
            lines += [
                self.orderby_clause.compile(),
            ]

        return LN.join(lines)


class SelectClause(ASTNode):
    def __init__(self, columns):
        self.columns = ColumnsClauseParameters(columns)

    def compile(self) -> str:
        parameters = self.columns.compile()
        return LN.join([SELECT, indent(parameters, prefix=SPC4)])


class FromClause(ASTNode):
    def __init__(self, tables, use_alias=True):
        # Remove duplicate tables, SQL doesn't accept FROM table1, table1
        self.tables = self._get_distinct_tables(tables)
        self.use_alias = use_alias

    def compile(self) -> str:
        parameters = SEPLN.join(
            table.compile(use_alias=self.use_alias) for table in self.tables
        )
        return LN.join([FROM, indent(parameters, prefix=SPC4)])

    @staticmethod
    def _get_distinct_tables(tables):
        return {table.name: table for table in tables}.values()


class WhereClause(ASTNode):
    def __init__(self, clauses):
        self.clauses = clauses

    def compile(self) -> str:
        parameters = ANDLN.join(clause.compile() for clause in self.clauses)
        return LN.join([WHERE, indent(parameters, prefix=SPC4)])


class ColumnsClauseParameters(ASTNode):
    def __init__(self, elements, use_alias=True, newline=True):
        self.elements = elements
        self.use_alias = use_alias
        self.newline = newline

    def compile(self) -> str:
        sep = SEPLN if self.newline else SEP
        return sep.join(
            element.compile(use_alias=self.use_alias) for element in self.elements
        )


class GroupbyClause(ASTNode):
    def __init__(self, columns):
        self.columns = ColumnsClauseParameters(columns, use_alias=False)

    def compile(self) -> str:
        parameters = self.columns.compile()
        return LN.join([GROUP_BY, indent(parameters, prefix=SPC4)])


class OrderbyClause(ASTNode):
    def __init__(self, columns):
        self.columns = columns

    def compile(self) -> str:
        parameters = SEPLN.join(
            column.compile(use_alias=False, use_prefix=False) for column in self.columns
        )
        return LN.join([ORDER_BY, indent(parameters, prefix=SPC4)])


class Insert(ASTNode):
    def __init__(self, table, values):
        self.table = table
        self.values = values

    def compile(self) -> str:
        if isinstance(self.values, Select):
            values = self.values.compile()
        else:
            raise NotImplementedError("Insert only accepts a nested Select")
        return LN.join([f"INSERT INTO {self.table}", values + ";"])


class CreateTable(ASTNode):
    def __init__(self, table: str, schema: List[Tuple[str, DType]]):
        self.table = table
        self.schema = schema

    def compile(self) -> str:
        schema = ",".join(f'"{name}" {dtype.to_sql()}' for name, dtype in self.schema)
        return f"CREATE TABLE {self.table}({schema});"


class Join(ASTNode):
    def __init__(self, left, right, l_alias, r_alias, on, type):
        self.left = left
        self.right = right
        self.l_alias = l_alias
        self.r_alias = r_alias
        self.on = on
        type = type.upper()
        if type not in ("INNER", "OUTER"):
            raise ValueError(f"Expected INNER or OUTER, got {type}.")
        self.type = type

    def compile(self):
        result = f"({self.left.compile()})" + " AS " + self.l_alias
        result += f"\n{self.type} JOIN\n"
        result += f"({self.right.compile()})" + " AS " + self.r_alias
        result += "\nON " + f"{self.l_alias}.{self.on}={self.r_alias}.{self.on}"
        return result
