"""
Lightweight cross-validation helpers adapted from the legacy flows.

These run locally (inside a UDF) and avoid repeated allocations where possible.
"""

from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import Optional
from typing import Tuple

import numpy as np


def kfold_indices(
    n_rows: int, n_splits: int
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate train/test indices for K-fold cross-validation.

    This is a small wrapper around numpy to avoid pulling in sklearn in UDFs.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_rows < n_splits:
        raise ValueError(
            f"Cannot split {n_rows} rows into {n_splits} folds (need n_rows >= n_splits)."
        )

    fold_sizes = np.full(n_splits, n_rows // n_splits, dtype=int)
    fold_sizes[: n_rows % n_splits] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = np.arange(start, stop)
        train_idx = np.concatenate((np.arange(0, start), np.arange(stop, n_rows)))
        yield train_idx, test_idx
        current = stop


def split_dataframe(df, n_splits: int) -> Iterable[Tuple["DataFrame", "DataFrame"]]:
    """
    Convenience wrapper: yield (train_df, test_df) pairs for each fold.
    """

    n_rows = len(df)
    for train_idx, test_idx in kfold_indices(n_rows, n_splits):
        yield df.iloc[train_idx], df.iloc[test_idx]


def min_rows_for_cv(
    df, y_var: str, n_splits: int, *, positive_class: Optional[object] = None
) -> Dict[str, object]:
    """
    Common per-worker check used by CV flows to ensure enough rows for splitting.
    """

    if y_var in df.columns:
        series = df[y_var]
        if positive_class is not None:
            series = series == positive_class
        n_obs = int(series.dropna().shape[0])
    else:
        n_obs = 0
    return {"ok": bool(n_obs >= int(n_splits)), "n_obs": n_obs}


def buffered_kfold_split(X: np.ndarray, y: np.ndarray, n_splits: int):
    """
    Yield train/test splits using reusable buffers to avoid per-fold allocations.
    """
    from sklearn.model_selection import KFold

    n_rows = X.shape[0]
    if n_rows == 0 or n_rows < n_splits:
        return

    kf = KFold(n_splits=int(n_splits), shuffle=False)

    train_X_buf = np.empty_like(X)
    test_X_buf = np.empty_like(X)
    train_y_buf = np.empty_like(y)
    test_y_buf = np.empty_like(y)

    for train_idx, test_idx in kf.split(np.arange(n_rows)):
        train_len = len(train_idx)
        test_len = len(test_idx)

        np.take(X, train_idx, axis=0, out=train_X_buf[:train_len])
        np.take(y, train_idx, axis=0, out=train_y_buf[:train_len])
        X_train = train_X_buf[:train_len, ...]
        y_train = train_y_buf[:train_len, ...]

        np.take(X, test_idx, axis=0, out=test_X_buf[:test_len])
        np.take(y, test_idx, axis=0, out=test_y_buf[:test_len])
        X_test = test_X_buf[:test_len, ...]
        y_test = test_y_buf[:test_len, ...]

        yield X_train, y_train, X_test, y_test
