from typing import List

from pydantic import BaseModel

from exareme2.algorithms.exaflow._metadata_utils import get_metadata_categories
from exareme2.algorithms.exaflow.algorithm import Algorithm
from exareme2.algorithms.exaflow.exaflow_registry import exaflow_udf
from exareme2.algorithms.exaflow.library.group_comparisons._cross_tab_table import (
    CrossTabTable,
)
from exareme2.algorithms.exaflow.library.group_comparisons.chi_squared import ChiSquared
from exareme2.algorithms.specifications import AlgorithmName

ALGORITHM_NAME = AlgorithmName.EXAFLOW_CHI_SQUARED


class ChiSquaredResult(BaseModel):
    factor: str
    outcome: str
    chi_square: float
    p_value: float
    degrees_of_freedom: int
    expected: List[List[float]]
    observed: List[List[float]]
    factor_categories: List[str]
    outcome_categories: List[str]


class ChiSquaredAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        factor = self.inputdata.x[0]
        outcome = self.inputdata.y[0]

        factor_categories = get_metadata_categories(
            metadata, factor, context="chi-squared test"
        )
        outcome_categories = get_metadata_categories(
            metadata, outcome, context="chi-squared test"
        )

        local_results = self.engine.run_algorithm_udf(
            func=local_step,
            positional_args={
                "inputdata": self.inputdata.json(),
                "factor": factor,
                "outcome": outcome,
                "factor_categories": factor_categories,
                "outcome_categories": outcome_categories,
            },
        )

        result = local_results[0]
        if any(r != result for r in local_results[1:]):
            raise ValueError("Worker results do not match")

        return ChiSquaredResult.parse_obj(result)


@exaflow_udf(with_aggregation_server=True)
def local_step(
    inputdata,
    csv_paths,
    agg_client,
    factor,
    outcome,
    factor_categories,
    outcome_categories,
):
    from exareme2.algorithms.utils.inputdata_utils import fetch_data

    data = fetch_data(inputdata, csv_paths)

    stat_func = ChiSquared(agg_client)
    chi_square, p_value, dof, expected = stat_func.compute(
        data,
        factor=factor,
        factor_categories=factor_categories,
        outcome=outcome,
        outcome_categories=outcome_categories,
    )

    cross_tab = CrossTabTable(agg_client).compute(
        data,
        factor=factor,
        factor_categories=factor_categories,
        outcome=outcome,
        outcome_categories=outcome_categories,
    )

    return {
        "factor": factor,
        "outcome": outcome,
        "chi_square": float(chi_square),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "expected": expected.astype(float).tolist(),
        "observed": cross_tab.astype(float).values.tolist(),
        "factor_categories": list(factor_categories),
        "outcome_categories": list(outcome_categories),
    }
