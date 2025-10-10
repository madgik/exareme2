from scipy.stats import fisher_exact

from exareme2.algorithms.exaflow.library.group_comparisons._cross_tab_table import (
    CrossTabTable,
)
from exareme2.algorithms.exaflow.library.templates.statistical_function import (
    StatisticalFunction,
)


class FisherExact(StatisticalFunction):
    def compute(
        self,
        dataset,
        *,
        factor,
        factor_categories,
        outcome,
        outcome_categories,
    ):
        cross_tab_table = (
            CrossTabTable(self.client)
            .compute(
                dataset,
                factor=factor,
                factor_categories=factor_categories,
                outcome=outcome,
                outcome_categories=outcome_categories,
            )
            .astype(float)
        )

        non_zero_rows = [
            idx for idx, total in cross_tab_table.sum(axis=1).items() if total > 0
        ]
        non_zero_cols = [
            idx for idx, total in cross_tab_table.sum(axis=0).items() if total > 0
        ]

        filtered_table = cross_tab_table.loc[non_zero_rows, non_zero_cols]

        if filtered_table.shape != (2, 2):
            raise ValueError(
                "Fisher's exact test requires exactly two non-empty categories for both factor and outcome."
            )

        odds_ratio, p_value = fisher_exact(filtered_table.values)
        return odds_ratio, p_value, filtered_table, non_zero_rows, non_zero_cols
