from copy import deepcopy
from typing import List

from mipengine import DType
from mipengine.exceptions import BadUserInput
from mipengine.udfgen import AdhocUdfGenerator
from mipengine.udfgen.udfgen_DTOs import UDFGenTableResult


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
