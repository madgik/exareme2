from __future__ import annotations

from typing import Dict
from typing import List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared pure helpers
# ---------------------------------------------------------------------------


def construct_design_labels(
    categorical_vars: List[str],
    dummy_categories: Dict[str, List],
    numerical_vars: List[str],
) -> List[str]:
    labels = ["Intercept"]
    for var in categorical_vars:
        labels.extend([f"{var}[{lvl}]" for lvl in dummy_categories.get(var, [])])
    labels.extend(numerical_vars)
    return labels


def build_design_matrix(
    data: pd.DataFrame,
    *,
    categorical_vars: List[str],
    dummy_categories: Dict[str, List],
    numerical_vars: List[str],
) -> np.ndarray:
    n_rows = len(data)
    columns = [np.ones((n_rows, 1), dtype=float)]  # Intercept

    # Categorical → dummy columns
    for var in categorical_vars:
        categories = dummy_categories.get(var, [])
        if var not in data.columns:
            columns.extend([np.zeros((n_rows, 1), dtype=float) for _ in categories])
            continue

        col = data[var]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]

        values = col
        for category in categories:
            encoded = (values == category).astype(float).to_numpy().reshape(-1, 1)
            columns.append(encoded)

    # Numerical vars
    for var in numerical_vars:
        if var not in data.columns:
            columns.append(np.zeros((n_rows, 1), dtype=float))
            continue

        col = data[var]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]

        num_col = col.astype(float).to_numpy().reshape(-1, 1)
        columns.append(num_col)

    return np.hstack(columns) if columns else np.empty((n_rows, 0), dtype=float)


def collect_categorical_levels_from_df(
    data: pd.DataFrame, categorical_vars: List[str]
) -> Dict[str, List]:
    """
    Core logic to compute observed levels per categorical variable
    from a local pandas DataFrame. This is what runs *inside* UDFs.
    """
    levels: Dict[str, List] = {}
    for var in categorical_vars:
        if var not in data.columns:
            levels[var] = []
            continue

        col = data[var]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]

        vals = col.dropna().unique().tolist()
        levels[var] = vals
    return levels


def get_dummy_categories(
    engine,
    inputdata_json: str,
    categorical_vars: List[str],
    collect_udf,
    *,
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
        "inputdata": inputdata_json,
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
