from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
from pydantic import BaseModel

from exaflow.algorithms.exareme3.algorithm import Algorithm
from exaflow.algorithms.exareme3.exaflow_registry import exaflow_udf
from exaflow.algorithms.exareme3.metadata_utils import validate_metadata_vars
from exaflow.algorithms.exareme3.validation_utils import require_dependent_var

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


class MultipleHistogramsAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    def run(self, metadata):
        require_dependent_var(
            self.inputdata,
            message="Multiple histograms requires a target variable in 'y' (grouping 'x' is optional).",
        )
        y_var = self.inputdata.y[0]
        x_vars = self.inputdata.x or []
        bins_param = self.parameters.get("bins", 20) or 20
        validate_metadata_vars([y_var] + x_vars, metadata)

        metadata_subset = {var: metadata[var] for var in {y_var, *x_vars}}

        results = self.engine.run_algorithm_udf(
            func=local_step,
            positional_args={
                "inputdata": self.inputdata.json(),
                "metadata": metadata_subset,
                "bins": int(bins_param),
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


def _value_counts(series, categories):
    counts = series.value_counts()
    return [int(counts.get(cat, 0)) for cat in categories]


def _get_enumerations(meta_entry, series):
    if meta_entry and meta_entry.get("enumerations"):
        return list(meta_entry["enumerations"].keys())
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


@exaflow_udf(with_aggregation_server=True)
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
    categories = _get_enumerations(metadata.get(y_var), data[y_var])
    local_counts = _value_counts(data[y_var], categories)
    global_counts = agg_client.sum(local_counts)
    masked_counts = _mask_counts(global_counts, min_row_count)

    grouped: Dict[str, Dict[str, List]] = {}
    for x_var in x_vars:
        groups = _get_enumerations(metadata.get(x_var), data[x_var])
        matrix = [
            _value_counts(data.loc[data[x_var] == group, y_var], categories)
            for group in groups
        ]
        global_matrix = _aggregate_matrix(agg_client, matrix)
        grouped[x_var] = {
            "groups": groups,
            "counts": [_mask_counts(row, min_row_count) for row in global_matrix],
        }

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

    return {
        "bins": bin_edges.tolist(),
        "counts": masked_counts,
        "grouped": grouped,
    }


def _value_counts(series, categories):
    counts = series.value_counts()
    return [int(counts.get(cat, 0)) for cat in categories]


def _get_enumerations(meta_entry, series):
    if meta_entry and meta_entry.get("enumerations"):
        return list(meta_entry["enumerations"].keys())
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
