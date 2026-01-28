"""
Preprocessing helpers kept minimal for migrated flows.
"""

from typing import Dict
from typing import List


def get_dummy_categories(
    *,
    engine,
    categorical_vars: List[str],
    collect_udf,
    extra_args: Dict | None = None,
) -> Dict[str, List]:
    """
    Discover dummy categories from the actual data (like DummyEncoder):

    - Runs the provided UDF (collect_udf) on each worker to get observed levels.
    - Merges and sorts.
    - Drops first level per variable (reference category).
    """
    if not categorical_vars:
        return {}

    positional_args = {
        "categorical_vars": categorical_vars,
    }
    if extra_args:
        positional_args.update(extra_args)

    worker_levels = engine.run_algorithm_udf(
        func=collect_udf,
        positional_args=positional_args,
    )

    merged = {var: set() for var in categorical_vars}
    for worker_result in worker_levels:
        for var, levels in worker_result.items():
            merged[var].update(level for level in levels if level is not None)

    sorted_levels = {var: sorted(merged.get(var, set())) for var in categorical_vars}

    # Drop first level to avoid multicollinearity and match DummyEncoder
    return {var: levels[1:] for var, levels in sorted_levels.items()}
