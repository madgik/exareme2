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
    # Preallocate once: intercept + dummies + numeric columns
    n_dummy_cols = sum(len(dummy_categories.get(var, [])) for var in categorical_vars)
    total_cols = 1 + n_dummy_cols + len(numerical_vars)
    design = np.empty((n_rows, total_cols), dtype=float)

    col_idx = 0
    design[:, col_idx] = 1.0  # Intercept
    col_idx += 1

    # Categorical → dummy columns
    for var in categorical_vars:
        categories = dummy_categories.get(var, [])
        if var not in data.columns:
            if categories:
                design[:, col_idx : col_idx + len(categories)] = 0.0
                col_idx += len(categories)
            continue

        col = data[var]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        values = col
        for category in categories:
            encoded = (values == category).to_numpy(dtype=float, copy=False).reshape(-1)
            design[:, col_idx] = encoded
            col_idx += 1

    # Numerical vars
    for var in numerical_vars:
        if var not in data.columns:
            design[:, col_idx] = 0.0
            col_idx += 1
            continue

        col = data[var]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]

        num_col = col.to_numpy(dtype=float, copy=False).reshape(-1)
        design[:, col_idx] = num_col
        col_idx += 1

    return design


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
