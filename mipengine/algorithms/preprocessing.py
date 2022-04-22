import json

from mipengine.udfgen import literal
from mipengine.udfgen import merge_transfer
from mipengine.udfgen import relation
from mipengine.udfgen import tensor
from mipengine.udfgen import transfer
from mipengine.udfgen import udf


class DesignMatrixPreprocessor:
    def __init__(self, executor, intercept=True):
        self._local_run = executor.run_udf_on_local_nodes
        self._global_run = executor.run_udf_on_global_node
        self._categorical_vars = [
            varname
            for varname in executor.x_variables
            if executor.metadata[varname]["is_categorical"]
        ]
        self._numerical_vars = [
            varname
            for varname in executor.x_variables
            if not executor.metadata[varname]["is_categorical"]
        ]
        self.intercept = intercept
        self.new_varnames = None

    def _gather_enums(self, x):
        if self._categorical_vars:
            local_transfers = self._local_run(
                func=self._gather_enums_local,
                keyword_args={"x": x, "categorical_vars": self._categorical_vars},
                share_to_global=[True],
            )
            global_transfer = self._global_run(
                func=self._gather_enums_global,
                keyword_args=dict(local_transfers=local_transfers),
                share_to_locals=[True],
            )
            enums = json.loads(global_transfer.get_table_data()[1][0])
            enums = {varname: sorted(e)[1:] for varname, e in enums.items()}
            enums = {
                varname: [
                    {
                        "code": code,
                        "dummy": f"{varname}__{i}",
                        "label": f"{varname}[{code}]",
                    }
                    for i, code in enumerate(e)
                ]
                for varname, e in enums.items()
            }
            return enums
        return {}

    @staticmethod
    @udf(x=relation(), categorical_vars=literal(), return_type=transfer())
    def _gather_enums_local(x, categorical_vars):
        categorical_vars = ["x_" + varname for varname in categorical_vars]
        enumerations = {}
        for cat in categorical_vars:
            enumerations[cat] = list(x[cat].unique())

        transfer_ = dict(enumerations=enumerations)
        return transfer_

    @staticmethod
    @udf(local_transfers=merge_transfer(), return_type=transfer())
    def _gather_enums_global(local_transfers):
        from functools import reduce

        def reduce_enums(varname):
            return list(
                reduce(
                    lambda a, b: set(a) | set(b),
                    [l["enumerations"][varname] for l in local_transfers],
                    {},
                )
            )

        keys = local_transfers[0]["enumerations"].keys()
        enumerations = {key[2:]: reduce_enums(key) for key in keys}

        return enumerations

    def _create_design_matrix(self, x, enums):
        design_matrix = self._local_run(
            func="create_design_matrix",
            keyword_args=dict(
                x=x,
                enums=enums,
                numerical_vars=self._numerical_vars,
                intercept=self.intercept,
            ),
            share_to_global=[False],
        )
        return design_matrix

    def _get_new_variable_names(self, numerical_vars, enums):
        names = []
        if self.intercept:
            names.append("Intercept")
        names.extend([varname for varname in numerical_vars])
        names.extend([e["label"] for enum in enums.values() for e in enum])
        return names

    def transform(self, x):
        enums = self._gather_enums(x)
        self.new_varnames = self._get_new_variable_names(self._numerical_vars, enums)
        if self._categorical_vars or self.intercept:
            x = self._create_design_matrix(x, enums)
        return x


def relation_to_vector(rel, executor):
    return executor.run_udf_on_local_nodes(
        func=relation_to_vector_local_udf,
        keyword_args={"rel": rel},
        share_to_global=[False],
    )


@udf(rel=relation(), return_type=tensor(dtype=float, ndims=1))
def relation_to_vector_local_udf(rel):
    return rel
