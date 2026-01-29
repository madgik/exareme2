from __future__ import annotations

from typing import Any

import pandas as pd

try:  # Optional dependency for workers that load Arrow tables
    import pyarrow as pa
except ImportError:  # pragma: no cover - pyarrow not available
    pa = None  # type: ignore[assignment]


def convert_to_pandas_dataframe(data: Any) -> pd.DataFrame:
    """
    Return a pandas DataFrame regardless of whether ``data`` is already a DataFrame,
    a PyArrow table, or any object exposing ``to_pandas``.
    """
    if isinstance(data, pd.DataFrame):
        return data

    if pa is not None and isinstance(data, pa.Table):
        return data.to_pandas()

    to_pandas = getattr(data, "to_pandas", None)
    if callable(to_pandas):
        df = to_pandas()
        if isinstance(df, pd.DataFrame):
            return df

    # Fallback: best-effort DataFrame construction
    return pd.DataFrame(data)
