from __future__ import annotations

from copy import deepcopy
from typing import Dict
from typing import List

import pandas as pd

from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.worker_communication import BadUserInput


def _validate_strategies(
    raw_x: List[str], raw_y: List[str], strategies: Dict[str, str]
):
    requested_vars = set(raw_x + raw_y)
    missing_or_extra = set(strategies.keys()) ^ requested_vars
    if missing_or_extra:
        raise BadUserInput(
            "A strategy must be provided for every variable in x and y, "
            "and only for those variables."
        )


def _validate_diff_not_nominal(strategies: Dict[str, str], metadata: Dict[str, dict]):
    for name, strategy in strategies.items():
        if strategy == "diff" and metadata.get(name, {}).get("is_categorical"):
            raise BadUserInput(
                f"Cannot take the difference for the nominal variable '{name}'."
            )


def _rename_variables(raw_vars: List[str], strategies: Dict[str, str]) -> List[str]:
    return [
        f"{name}_diff" if strategies.get(name) == "diff" else name for name in raw_vars
    ]


def prepare_longitudinal_transformation(
    inputdata: Inputdata, metadata: Dict[str, dict], params: Dict[str, object]
) -> tuple[Inputdata, Dict[str, dict], Dict[str, object]]:
    visit1 = params.get("visit1")
    visit2 = params.get("visit2")
    strategies = params.get("strategies", {})
    raw_x = inputdata.x or []
    raw_y = inputdata.y or []

    if not visit1 or not visit2:
        raise BadUserInput("Both 'visit1' and 'visit2' parameters are required.")

    _validate_strategies(raw_x, raw_y, strategies)
    _validate_diff_not_nominal(strategies, metadata)

    transformed_x = _rename_variables(raw_x, strategies)
    transformed_y = _rename_variables(raw_y, strategies)
    transformed_inputdata = inputdata.copy(
        update={"x": transformed_x, "y": transformed_y}
    )

    transformed_metadata = deepcopy(metadata)
    for varname, strategy in strategies.items():
        if strategy == "diff":
            transformed_metadata[f"{varname}_diff"] = transformed_metadata.pop(varname)

    preprocessing_payload = {
        "visit1": visit1,
        "visit2": visit2,
        "strategies": strategies,
        "raw_x": raw_x,
        "raw_y": raw_y,
        "transformed_x": transformed_x,
        "transformed_y": transformed_y,
    }

    return transformed_inputdata, transformed_metadata, preprocessing_payload


def _build_empty_result(columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def apply_longitudinal_transformation(
    df: pd.DataFrame, payload: Dict[str, object]
) -> pd.DataFrame:
    visit1 = payload["visit1"]
    visit2 = payload["visit2"]
    strategies: Dict[str, str] = payload["strategies"]  # type: ignore[assignment]
    raw_x: List[str] = payload.get("raw_x", [])  # type: ignore[assignment]
    raw_y: List[str] = payload.get("raw_y", [])  # type: ignore[assignment]
    transformed_x: List[str] = payload.get("transformed_x", [])  # type: ignore[assignment]
    transformed_y: List[str] = payload.get("transformed_y", [])  # type: ignore[assignment]

    required_columns = set(raw_x + raw_y + ["subjectid", "visitid"])
    missing = required_columns - set(df.columns)
    if missing:
        raise BadUserInput(
            f"Missing required columns for longitudinal transformation: {sorted(missing)}"
        )

    df = df[df["visitid"].isin([visit1, visit2])]
    if df.empty:
        return _build_empty_result(transformed_x + transformed_y)

    key_cols = ["subjectid"]
    if "dataset" in df.columns:
        key_cols.append("dataset")

    left = df[df["visitid"] == visit1]
    right = df[df["visitid"] == visit2]
    merged = left.merge(right, on=key_cols, suffixes=("_v1", "_v2"), how="inner")
    if merged.empty:
        return _build_empty_result(transformed_x + transformed_y)

    result = merged[key_cols].copy()

    for varname, strategy in strategies.items():
        v1 = merged[f"{varname}_v1"]
        v2 = merged[f"{varname}_v2"]
        if strategy == "first":
            result[varname] = v1
        elif strategy == "second":
            result[varname] = v2
        elif strategy == "diff":
            result[f"{varname}_diff"] = v2 - v1
        else:
            raise BadUserInput(
                f"Unknown strategy '{strategy}' for variable '{varname}'."
            )

    # Keep only the transformed columns (plus any keys) to mirror the requested inputdata.
    desired_columns = []
    for col in key_cols + transformed_x + transformed_y:
        if col not in desired_columns:
            desired_columns.append(col)
    return result[[col for col in desired_columns if col in result.columns]]
