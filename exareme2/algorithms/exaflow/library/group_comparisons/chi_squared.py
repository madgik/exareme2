from scipy.stats import chi2_contingency

from exareme2.algorithms.exaflow.library.group_comparisons._cross_tab_table import (
    CrossTabTable,
)
from exareme2.algorithms.exaflow.library.templates.statistical_function import (
    StatisticalFunction,
)


class ChiSquared(StatisticalFunction):
    def compute(
        self,
        dataset,
        *,
        factor,
        factor_categories,
        outcome,
        outcome_categories,
    ):
        cross_tab_table = CrossTabTable(self.client).compute(
            dataset,
            factor=factor,
            factor_categories=factor_categories,
            outcome=outcome,
            outcome_categories=outcome_categories,
        )
        chi2, p_value, dof, expected = chi2_contingency(cross_tab_table)
        return chi2, p_value, dof, expected
