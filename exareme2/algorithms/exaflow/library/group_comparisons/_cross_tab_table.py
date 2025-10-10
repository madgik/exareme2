import pandas as pd

from exareme2.algorithms.exaflow.library.templates.statistical_function import (
    StatisticalFunction,
)


class CrossTabTable(StatisticalFunction):
    def compute(
        self,
        dataset,
        *,
        factor,
        factor_categories,
        outcome,
        outcome_categories,
    ):
        dataset[factor] = pd.Categorical(dataset[factor], categories=factor_categories)
        dataset[outcome] = pd.Categorical(
            dataset[outcome], categories=outcome_categories
        )
        cross_tab = pd.crosstab(
            dataset[factor],
            dataset[outcome],
            dropna=False,
        )

        aggregated = self.client.fed_sum(cross_tab.values)
        cross_tab.iloc[:] = aggregated.astype(float)

        return cross_tab
