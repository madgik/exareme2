import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
import pandas as pd
from pandas.api import types as pd_types
from pydantic import BaseModel

from exaflow.algorithms.exareme3.algorithm import Algorithm
from exaflow.algorithms.exareme3.exareme3_registry import exareme3_udf

HistogramBin = Union[float, str]


class Histogram(BaseModel):
    var: str
    grouping_var: Optional[str]
    grouping_enum: Optional[str]
    bins: List[HistogramBin]
    counts: List[Optional[int]]


class HistogramResult(BaseModel):
    histogram: List[Histogram]


ALGORITHM_NAME = "multiple_histograms"

logger = logging.getLogger(__name__)


class MultipleHistogramsAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        y_var = self.inputdata.y[0]
        x_vars = self.inputdata.x or []

        default_bins = 20
        bins = self.parameters.get("bins", default_bins)
        if bins is None:
            bins = default_bins

        metadata_subset = {var: metadata[var] for var in {y_var, *x_vars}}

        results = self.engine.run_algorithm_udf(
            func=local_step,
            positional_args={
                "inputdata": self.inputdata.json(),
                "metadata": metadata_subset,
                "bins": bins,
            },
        )
        payload = results[0]

        histograms: List[Histogram] = []
        base_bins = payload["bins"]
        histograms.append(
            Histogram(
                var=y_var,
                grouping_var=None,
                grouping_enum=None,
                bins=base_bins,
                counts=payload["counts"],
            )
        )

        for grouping_var, grouped in payload.get("grouped", {}).items():
            groups = grouped["groups"]
            counts_per_group = grouped["counts"]
            for group, counts in zip(groups, counts_per_group):
                histograms.append(
                    Histogram(
                        var=y_var,
                        grouping_var=grouping_var,
                        grouping_enum=group,
                        bins=base_bins,
                        counts=counts,
                    )
                )

        return HistogramResult(histogram=histograms)


def _value_counts(series: pd.Series, categories, *, label: Optional[str] = None):
    counts = series.value_counts()
    series_label = label or "series"
    dtype = series.dtype
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "value_counts start label=%s dtype=%s categories=%s distinct_values=%s",
            series_label,
            dtype,
            categories,
            counts.index.tolist(),
        )
    resolved = []
    for cat in categories:
        count = counts.get(cat, 0)
        if count == 0 and isinstance(cat, str):
            fallback = _value_counts_numeric_fallback(counts, cat, series_label)
            if fallback:
                logger.debug(
                    "value_counts fallback hit label=%s cat=%s dtype=%s fallback=%s",
                    series_label,
                    cat,
                    dtype,
                    fallback,
                )
            count = fallback
        if count == 0 and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "value_counts zero bin label=%s cat=%s dtype=%s available_keys=%s",
                series_label,
                cat,
                dtype,
                counts.index.tolist(),
            )
        resolved.append(int(count))
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "value_counts result label=%s dtype=%s resolved=%s",
            series_label,
            dtype,
            resolved,
        )
    return resolved


def _value_counts_numeric_fallback(counts, cat: str, label: str) -> int:
    for caster in (float, int):
        try:
            coerced = caster(cat)
        except (TypeError, ValueError):
            continue
        value = counts.get(coerced, 0)
        logger.debug(
            "value_counts numeric fallback attempt label=%s cat=%s caster=%s coerced=%s value=%s",
            label,
            cat,
            caster.__name__,
            coerced,
            value,
        )
        if value:
            return value
    return 0


def _normalize_categories(categories: List[HistogramBin], series: pd.Series):
    dtype = series.dtype
    if pd_types.is_integer_dtype(dtype):

        def caster(value):
            return int(float(value))

    elif pd_types.is_float_dtype(dtype):

        def caster(value):
            return float(value)

    else:
        return categories

    normalized: List[HistogramBin] = []
    for value in categories:
        if value is None:
            normalized.append(value)
            continue
        try:
            normalized.append(caster(value))
        except (TypeError, ValueError):
            normalized.append(value)
    return normalized


def _get_enumerations(meta_entry, series: pd.Series):
    if meta_entry and meta_entry.get("enumerations"):
        categories = list(meta_entry["enumerations"].keys())
        return _normalize_categories(categories, series)
    return sorted(series.dropna().unique().tolist())


def _mask_counts(values: Sequence[float], min_row_count: int) -> List[Optional[int]]:
    masked = []
    for value in values:
        count = int(round(value))
        masked.append(count if count >= min_row_count else None)
    return masked


def _aggregate_matrix(agg_client, matrix: Sequence[Sequence[int]]):
    if not matrix:
        return []
    arr = np.asarray(matrix, dtype=float)
    if arr.size == 0:
        return arr.tolist()
    flat = arr.reshape(-1).tolist()
    summed = agg_client.sum(flat)
    return np.asarray(summed).reshape(arr.shape).tolist()


@exareme3_udf(with_aggregation_server=True)
def local_step(data, inputdata, agg_client, metadata, bins):
    from exaflow.worker import config as worker_config

    min_row_count = worker_config.privacy.minimum_row_count

    y_var = inputdata.y[0]
    x_vars = inputdata.x or []

    y_meta = metadata.get(y_var, {})
    is_categorical = y_meta.get("is_categorical", False)

    if is_categorical:
        return _categorical_histogram(
            data=data,
            y_var=y_var,
            x_vars=x_vars,
            metadata=metadata,
            agg_client=agg_client,
            min_row_count=min_row_count,
        )

    return _numerical_histogram(
        data=data,
        y_var=y_var,
        x_vars=x_vars,
        metadata=metadata,
        agg_client=agg_client,
        bins=bins,
        min_row_count=min_row_count,
    )


def _categorical_histogram(data, y_var, x_vars, metadata, agg_client, min_row_count):
    series = data[y_var]
    categories = _get_enumerations(metadata.get(y_var), series)
    local_counts = _value_counts(series, categories, label=f"{y_var}")
    global_counts = agg_client.sum(local_counts)
    masked_counts = _mask_counts(global_counts, min_row_count)
    if all(count is None for count in masked_counts):
        logger.warning(
            "Histogram counts fully masked for var='%s'. dtype=%s categories=%s local=%s global=%s min_row_count=%s",
            y_var,
            series.dtype,
            categories,
            local_counts,
            global_counts,
            min_row_count,
        )
    else:
        logger.debug(
            "Histogram counts for var='%s'. dtype=%s categories=%s local=%s global=%s masked=%s min_row_count=%s",
            y_var,
            series.dtype,
            categories,
            local_counts,
            global_counts,
            masked_counts,
            min_row_count,
        )

    grouped: Dict[str, Dict[str, List]] = {}
    for x_var in x_vars:
        x_series = data[x_var]
        groups = _get_enumerations(metadata.get(x_var), x_series)
        matrix = [
            _value_counts(
                data.loc[x_series == group, y_var],
                categories,
                label=f"{y_var}|{x_var}=={group}",
            )
            for group in groups
        ]
        global_matrix = _aggregate_matrix(agg_client, matrix)
        grouped[x_var] = {
            "groups": groups,
            "counts": [_mask_counts(row, min_row_count) for row in global_matrix],
        }
        logger.debug(
            "Grouped histogram for y='%s' by x='%s'. groups=%s local_matrix=%s global_matrix=%s",
            y_var,
            x_var,
            groups,
            matrix,
            global_matrix,
        )

    return {
        "bins": categories,
        "counts": masked_counts,
        "grouped": grouped,
    }


def _numerical_histogram(
    data,
    y_var,
    x_vars,
    metadata,
    agg_client,
    bins,
    min_row_count,
):
    import numpy as np

    bins = max(1, int(round(bins)))
    values = data[y_var].to_numpy(dtype=float, copy=False)
    n_obs = int(values.size)
    total_n_obs_arr = agg_client.sum(np.array([float(n_obs)], dtype=float))
    global_min_arr = agg_client.min(
        np.array([float(np.min(values)) if n_obs else np.inf], dtype=float)
    )
    global_max_arr = agg_client.max(
        np.array([float(np.max(values)) if n_obs else -np.inf], dtype=float)
    )
    total_n_obs = total_n_obs_arr[0]
    if total_n_obs == 0:
        raise ValueError("No data available to compute histogram.")

    global_min = float(global_min_arr[0])
    global_max = float(global_max_arr[0])
    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise ValueError("Unable to determine histogram bounds.")
    if global_min == global_max:
        global_max = global_min + 1.0

    bin_edges = np.linspace(global_min, global_max, bins + 1)
    local_hist, _ = np.histogram(values, bins=bin_edges)
    global_hist = agg_client.sum(local_hist)
    masked_counts = _mask_counts(global_hist, min_row_count)
    if all(count is None for count in masked_counts):
        logger.warning(
            "Numerical histogram fully masked for var='%s'. total_rows=%s global_hist=%s min_row_count=%s bin_edges=%s",
            y_var,
            total_n_obs,
            global_hist,
            min_row_count,
            bin_edges.tolist(),
        )
    else:
        logger.debug(
            "Numerical histogram for var='%s'. total_rows=%s global_hist=%s masked=%s min_row_count=%s bin_edges=%s",
            y_var,
            total_n_obs,
            global_hist,
            masked_counts,
            min_row_count,
            bin_edges.tolist(),
        )

    grouped: Dict[str, Dict[str, List]] = {}
    for x_var in x_vars:
        groups = _get_enumerations(metadata.get(x_var), data[x_var])
        matrix = []
        for group in groups:
            subset = data.loc[data[x_var] == group, y_var].to_numpy(
                dtype=float, copy=False
            )
            group_hist, _ = np.histogram(subset, bins=bin_edges)
            matrix.append(group_hist.tolist())
        global_matrix = _aggregate_matrix(agg_client, matrix)
        grouped[x_var] = {
            "groups": groups,
            "counts": [_mask_counts(row, min_row_count) for row in global_matrix],
        }
        logger.debug(
            "Grouped numerical histogram for y='%s' by x='%s'. groups=%s global_matrix=%s",
            y_var,
            x_var,
            groups,
            global_matrix,
        )

    return {
        "bins": bin_edges.tolist(),
        "counts": masked_counts,
        "grouped": grouped,
    }
