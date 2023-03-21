from abc import ABC
from abc import abstractproperty
from types import SimpleNamespace
from typing import Dict
from typing import List
from typing import Tuple

from mipengine import DType
from mipengine.udfgen.ast import Column
from mipengine.udfgen.ast import ConstColumn
from mipengine.udfgen.ast import CreateTable
from mipengine.udfgen.ast import Insert
from mipengine.udfgen.ast import Join
from mipengine.udfgen.ast import Select
from mipengine.udfgen.ast import Table
from mipengine.udfgen.helpers import make_unique_func_name
from mipengine.udfgen.py_udfgenerator import FlowUdfArg
from mipengine.udfgen.udfgenerator import UdfGenerator


class AdhocUdfGenerator(UdfGenerator, ABC):
    """This abstract class can be subclassed to define ad hoc UDFs

    The user needs to define two abstract methods and one abstract property.

    `get_exec_stmt` should return a string with the statement needed for
    executing the UDF and storing its results into the return table.

    `get_results` should return a list of `UDFGenTableResult`. Currently, only
    a single result is supported. Each `UDFGenTableResult` is an object
    representing the output table. The user needs to provide three args to each
    `UDFGenTableResult`. The table name, its schema and its creation SQL
    statement.

    `output_schema` is a property representing the schema of the main output
    table.

    Additionally, the user can override `get_definition` to create ad hoc
    python UDFs. `get_definition` should return a string with the UDF
    definition. The default implementation returns an empty string for pure SQL
    UDFs.

    In order to facilitate the creation of SQL strings, a `ast` attribue is
    available containing the classes `Column`, `ConstColumn`, `Table`,
    `Select`, `Insert` and `CreateTable`. These all expose a `compile` method
    returning a string with the corresponding SQL construct.

    Examples
    --------
    >>> class MyUdfGen(AdhocUdfGenerator):
    ...     @property
    ...     def output_schema(self):
    ...         return [('col', DType.INT)]
    ...
    ...     def get_exec_stmt(self, udf_name, output_table_names):
    ...         table_name = output_table_names[0]
    ...         return f"INSERT INTO {table_name} SELECT 1"
    ...
    ...     def get_results(self, output_table_names):
    ...         table_name = output_table_names[0]
    ...         return UDFGenTableResult(
    ...             table_name=table_name,
    ...             table_schema=self.output_schema,
    ...             create_query=f"CREATE TABLE {table_name}(col INT);"
    ...         )
    """

    _registry = {}
    # AdhocUdfGenerator should always produce a single output
    num_outputs = 1

    # Make AST classes locally available to users
    ast = SimpleNamespace(
        Column=Column,
        ConstColumn=ConstColumn,
        Table=Table,
        Select=Select,
        Insert=Insert,
        CreateTable=CreateTable,
        Join=Join,
    )

    def __init__(
        self,
        flowkwargs: Dict[str, FlowUdfArg],
        smpc_used: bool = False,
        output_schema=None,
        min_row_count: int = None,
    ):
        """
        Parameters
        ----------
        flowkwargs : Dict[str, FlowUdfArg]
            Keyword arguments received by the algorithm flow
        smpc_used : bool
            True when SMPC framework is used
        output_schema : List[Tuple[str, DType]]
            This argument needs to be provided when the output schema is declared as
            deferred
        min_row_count : int
            Minimum allowed number of rows for an input table
        kwargs : dict
            Remaining UdfGenerator arguments which are not used in this class. This
            is necessary in order to match PyUdfGenerator's API.
        """
        # make flowkwargs accessible as attribues
        self.__dict__.update(flowkwargs)

        self._smpc_used = smpc_used
        self._output_schema = output_schema
        self._min_row_count = min_row_count

    @classmethod
    def __init_subclass__(cls, /, **kwargs):
        cls.fname = make_unique_func_name(cls)
        super().__init_subclass__(**kwargs)
        cls._registry[cls.fname] = cls

    @classmethod
    def is_registered(cls, funcname):
        return funcname in cls._registry

    @classmethod
    def get_subclass(cls, funcname):
        return cls._registry[funcname]

    @abstractproperty
    def output_schema(self) -> List[Tuple[str, DType]]:
        pass

    def get_definition(self, udf_name: str, output_table_names: List[str]) -> str:
        return ""
