from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from mipengine import DType
from mipengine.algorithm_specification import AlgorithmSpecification
from mipengine.exceptions import BadUserInput
from mipengine.udfgen import AdhocUdfGenerator
from mipengine.udfgen.udfgen_DTOs import UDFGenTableResult

if TYPE_CHECKING:
    from mipengine.controller.algorithm_execution_engine import AlgorithmExecutionEngine

TRANSFORMER_NAME = "longitudinal_transformer"


@dataclass(frozen=True)
class Result:
    data: tuple
    metadata: dict


@dataclass(frozen=True)
class Variables:
    x: List[str]
    y: List[str]


class DataLoader:
    def __init__(self, variables: Variables):
        self._variables = variables

    def get_variable_groups(self):
        xvars = self._variables.x
        yvars = self._variables.y
        xvars += ["subjectid", "visitid"]
        yvars += ["subjectid", "visitid"]
        return [xvars, yvars]

    def get_dropna(self) -> bool:
        return True

    def get_check_min_rows(self) -> bool:
        return True

    def get_variables(self) -> Variables:
        return self._variables


@dataclass(frozen=True)
class InitializationParams:
    datasets: List[str]
    var_filters: Optional[dict] = None
    algorithm_parameters: Optional[Dict[str, Any]] = None


class LongitudinalTransformerRunner:
    def __init__(
        self,
        initialization_params: InitializationParams,
        data_loader: DataLoader,
        engine: "AlgorithmExecutionEngine",
    ):
        self.algorithm_parameters = initialization_params.algorithm_parameters
        self.variables = data_loader.get_variables()
        self.engine = engine

    @classmethod
    def get_transformer_name(cls):
        return TRANSFORMER_NAME

    @classmethod
    def get_specification(cls):
        file = Path(__file__).parent / f"{cls.get_transformer_name()}.json"
        return AlgorithmSpecification.parse_file(file)

    def run(self, data, metadata):
        X, y = data
        metadata: dict = metadata

        visit1 = self.algorithm_parameters["visit1"]
        visit2 = self.algorithm_parameters["visit2"]
        strategies = self.algorithm_parameters["strategies"]

        # Split strategies to X and Y part
        x_strats = {
            name: strat
            for name, strat in strategies.items()
            if name in self.variables.x
        }
        y_strats = {
            name: strat
            for name, strat in strategies.items()
            if name in self.variables.y
        }

        lt_x = LongitudinalTransformer(self.engine, metadata, x_strats, visit1, visit2)
        X = lt_x.transform(X)
        metadata = lt_x.transform_metadata(metadata)

        lt_y = LongitudinalTransformer(self.engine, metadata, y_strats, visit1, visit2)
        y = lt_y.transform(y)
        metadata = lt_y.transform_metadata(metadata)

        data = (X, y)

        return Result(data, metadata)


class LongitudinalTransformer:
    def __init__(self, engine, metadata, strategies, visit1, visit2):
        for name, strategy in strategies.items():
            if strategy == "diff" and metadata[name]["is_categorical"]:
                msg = f"Cannot take the difference for the nominal variable '{name}'."
                raise BadUserInput(msg)

        self._local_run = engine.run_udf_on_local_nodes
        self._global_run = engine.run_udf_on_global_node
        self.metadata = metadata
        self.strategies = strategies
        self.visit1 = visit1
        self.visit2 = visit2

    def transform(self, X):
        schema = self._make_output_schmema()
        return self._local_run(
            func=LongitudinalTransformerUdf,
            keyword_args={
                "dataframe": X,
                "visit1": self.visit1,
                "visit2": self.visit2,
                "strategies": self.strategies,
            },
            output_schema=schema,
        )

    def transform_metadata(self, metadata: dict) -> dict:
        metadata = deepcopy(metadata)
        for varname, strategy in self.strategies.items():
            if strategy == "diff":
                metadata[f"{varname}_diff"] = metadata.pop(varname)
        return metadata

    def _make_output_schmema(self):
        def format_name(name: str, strategy: str) -> str:
            return f"{name}_diff" if strategy == "diff" else name

        def make_dtype(name: str) -> DType:
            return DType.from_cde(self.metadata[name]["sql_type"])

        schema = [("row_id", DType.INT)]
        schema += [
            (format_name(name, strategy), make_dtype(name))
            for name, strategy in self.strategies.items()
        ]
        return schema


class LongitudinalTransformerUdf(AdhocUdfGenerator):
    @property
    def output_schema(self):
        return self._output_schema

    def get_definition(self, udf_name: str, output_table_names: List[str]) -> str:
        return ""

    def get_exec_stmt(self, udf_name: None, output_table_names: List[str]) -> str:
        table_name, *_ = output_table_names

        # Left subquery for join
        left_terms = [self.ast.Column("*")]
        left_select = self.ast.Select(
            columns=left_terms,
            from_=[self.ast.Table(name=self.dataframe.name, columns=left_terms)],
            where=[self.ast.Column("visitid") == self.visit1],
        )

        # Right subquery for join
        right_terms = [self.ast.Column("*")]
        right_select = self.ast.Select(
            columns=right_terms,
            from_=[self.ast.Table(name=self.dataframe.name, columns=right_terms)],
            where=[self.ast.Column("visitid") == self.visit2],
        )

        # Main select query
        join = self.ast.Join(
            left_select,
            right_select,
            l_alias="t1",
            r_alias="t2",
            on="subjectid",
            type="inner",
        )
        main_terms = [self.ast.Column("row_id", table="t1")]
        for colname, strategy in self.strategies.items():
            main_terms.append(self._term_from_strategy(strategy, colname))
        # Currently Select cannot accept a Join in the from_ arg, and it's
        # probably not worth the effort to make this change. Instead, I create
        # a table having the JOIN expression for its name.
        table = self.ast.Table(name=join.compile(), columns=main_terms)
        select_main = self.ast.Select(main_terms, from_=[table])

        return self.ast.Insert(table=table_name, values=select_main).compile()

    def get_results(self, output_table_names: List[str]) -> List[UDFGenTableResult]:
        table_name, *_ = output_table_names
        create_stmt = self.ast.CreateTable(table_name, self.output_schema).compile()
        return [
            UDFGenTableResult(
                table_name=table_name,
                table_schema=self.output_schema,
                create_query=create_stmt,
            )
        ]

    def _term_from_strategy(self, strategy, colname):
        if strategy == "first":
            return self.ast.Column(colname, "t1")
        elif strategy == "second":
            return self.ast.Column(colname, "t2")
        elif strategy == "diff":
            return self.ast.Column(colname, "t2") - self.ast.Column(colname, "t1")
        raise NotImplementedError
