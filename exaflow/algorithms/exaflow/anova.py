from typing import List
from typing import Optional

from pydantic import BaseModel

from exaflow.algorithms.exaflow.algorithm import Algorithm
from exaflow.algorithms.exaflow.exaflow_registry import exaflow_udf
from exaflow.worker_communication import BadUserInput

ALGORITHM_NAME = "anova"


class AnovaResult(BaseModel):
    terms: List[str]
    sum_sq: List[float]
    df: List[int]
    f_stat: List[Optional[float]]
    f_pvalue: List[Optional[float]]


class AnovaTwoWayAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        if not self.inputdata.y or not self.inputdata.x:
            raise BadUserInput(
                "ANOVA two-way requires exactly one y and two x variables."
            )

        xs = self.inputdata.x
        if len(xs) != 2:
            raise BadUserInput(
                f"Anova two-way only works with two independent variables. "
                f"Got {len(xs)} variable(s) instead."
            )

        x1, x2 = xs
        y = self.inputdata.y[0]

        sstype = self.parameters.get("sstype")
        if sstype not in (1, 2):
            raise BadUserInput("Parameter 'sstype' must be 1 or 2.")

        levels_a = list(metadata[x1]["enumerations"])
        levels_b = list(metadata[x2]["enumerations"])
        if len(levels_a) < 2:
            raise BadUserInput(
                f"The variable {x1} has less than 2 levels and Anova cannot be "
                "performed. Please choose another variable."
            )
        if len(levels_b) < 2:
            raise BadUserInput(
                f"The variable {x2} has less than 2 levels and Anova cannot be "
                "performed. Please choose another variable."
            )

        results = self.engine.run_algorithm_udf(
            func=anova_twoway_local_step,
            positional_args={
                "inputdata": self.inputdata.json(),
                "x1": x1,
                "x2": x2,
                "y": y,
                "levels_a": levels_a,
                "levels_b": levels_b,
                "sstype": sstype,
            },
        )
        metrics = results[0]

        return AnovaResult(
            terms=metrics["terms"],
            sum_sq=metrics["sum_sq"],
            df=metrics["df"],
            f_stat=metrics["f_stat"],
            f_pvalue=metrics["f_pvalue"],
        )


@exaflow_udf(with_aggregation_server=True)
def anova_twoway_local_step(
    data, inputdata, agg_client, x1, x2, y, levels_a, levels_b, sstype
):
    import numpy as np

    from exaflow.algorithms.exaflow.library.stats.stats import anova_twoway

    if data.empty or any(col not in data.columns for col in (x1, x2, y)):
        x_arr = np.empty((0, 2), dtype=float)
        y_arr = np.empty((0,), dtype=float)
    else:
        col_a = data[x1]
        col_b = data[x2]
        col_y = data[y].astype(float)

        x_arr = np.column_stack([col_a.to_numpy(), col_b.to_numpy()])
        y_arr = col_y.to_numpy()

    return anova_twoway(
        agg_client=agg_client,
        x=x_arr,
        y=y_arr,
        x1_name=x1,
        x2_name=x2,
        levels_a=levels_a,
        levels_b=levels_b,
        sstype=sstype,
    )
