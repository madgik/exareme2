from typing import List

from pydantic import BaseModel

from exareme2.algorithms.exaflow._metadata_utils import get_metadata_categories
from exareme2.algorithms.exaflow.algorithm import Algorithm
from exareme2.algorithms.exaflow.exaflow_registry import exaflow_udf
from exareme2.algorithms.exaflow.library.group_comparisons.fisher_exact import (
    FisherExact,
)
from exareme2.algorithms.specifications import AlgorithmName

ALGORITHM_NAME = AlgorithmName.EXAFLOW_FISHER_EXACT


class FisherExactResult(BaseModel):
    factor: str
    outcome: str
    odds_ratio: float
    p_value: float
    observed: List[List[float]]
    factor_categories: List[str]
    outcome_categories: List[str]


class FisherExactAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        factor = self.inputdata.x[0]
        outcome = self.inputdata.y[0]

        factor_categories = get_metadata_categories(
            metadata,
            factor,
            min_length=2,
            context="Fisher's exact test",
        )
        outcome_categories = get_metadata_categories(
            metadata,
            outcome,
            min_length=2,
            context="Fisher's exact test",
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

        return FisherExactResult.parse_obj(result)


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

    stat_func = FisherExact(agg_client)
    (
        odds_ratio,
        p_value,
        observed_table,
        used_factor_categories,
        used_outcome_categories,
    ) = stat_func.compute(
        data,
        factor=factor,
        factor_categories=factor_categories,
        outcome=outcome,
        outcome_categories=outcome_categories,
    )

    return {
        "factor": factor,
        "outcome": outcome,
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "observed": observed_table.values.tolist(),
        "factor_categories": used_factor_categories,
        "outcome_categories": used_outcome_categories,
    }
