import warnings
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set

import pandas as pd
from pydantic import BaseModel


class Inputdata(BaseModel):
    data_model: str
    datasets: List[str]
    validation_datasets: Optional[List[str]] = None
    filters: Optional[dict] = None
    y: Optional[List[str]] = None
    x: Optional[List[str]] = None


def _apply_filter(df: pd.DataFrame, filter_rule: Dict[str, Any]) -> pd.DataFrame:
    """
    Recursively applies filter rules to the dataframe.

    Args:
        df: The input DataFrame.
        filter_rule: A dictionary containing 'condition' ("AND"/"OR") and a list of 'rules'.

    Returns:
        The filtered DataFrame.
    """
    condition = filter_rule.get("condition", "AND")
    rules = filter_rule.get("rules", [])

    if condition == "AND":
        filtered_df = df
        for rule in rules:
            filtered_df = _apply_single_rule(filtered_df, rule)
        return filtered_df
    elif condition == "OR":
        filtered_dfs = []
        for rule in rules:
            filtered_dfs.append(_apply_single_rule(df, rule))
        return pd.concat(filtered_dfs).drop_duplicates().reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported condition: {condition}")


def _apply_single_rule(df: pd.DataFrame, rule: Dict[str, Any]) -> pd.DataFrame:
    """
    Applies a single filter rule to the dataframe. If the rule is composite (contains a nested condition),
    it will recursively apply the filter.

    Args:
        df: The input DataFrame.
        rule: A dictionary defining the filter rule.

    Returns:
        The DataFrame filtered according to the rule.
    """
    # If the rule is composite, delegate to _apply_filter.
    if "condition" in rule:
        return _apply_filter(df, rule)
    else:
        column = rule["id"]
        operator = rule["operator"]
        value = rule.get("value", None)

        operator_functions = {
            "equal": lambda df, col, value: df[df[col] == value],
            "not_equal": lambda df, col, value: df[df[col] != value],
            "greater": lambda df, col, value: df[df[col] > value],
            "less": lambda df, col, value: df[df[col] < value],
            "greater_or_equal": lambda df, col, value: df[df[col] >= value],
            "less_or_equal": lambda df, col, value: df[df[col] <= value],
            "between": lambda df, col, value: df[df[col].between(value[0], value[1])],
            "not_between": lambda df, col, value: df[
                ~df[col].between(value[0], value[1])
            ],
            "is_not_null": lambda df, col, value: df[df[col].notnull()],
            "is_null": lambda df, col, value: df[df[col].isnull()],
            "in": lambda df, col, value: df[df[col].isin(value)],
        }

        if operator in operator_functions:
            return operator_functions[operator](df, column, value)
        else:
            raise ValueError(f"Unsupported operator: {operator}")


def _apply_inputdata(df: pd.DataFrame, inputdata: Inputdata) -> pd.DataFrame:
    """
    Filters the dataframe based on the provided inputdata including filters, datasets, and selected columns.
    """
    if inputdata.filters:
        df = _apply_filter(df, inputdata.filters)

    # Filter based on the provided datasets.
    all_datasets = inputdata.datasets + (
        inputdata.validation_datasets if inputdata.validation_datasets else []
    )
    df = df[df["dataset"].isin(all_datasets)]

    x_columns = inputdata.x if inputdata.x is not None else []
    y_columns = inputdata.y if inputdata.y is not None else []
    columns = x_columns + y_columns

    if not columns:
        raise ValueError("Both 'x' and 'y' columns are missing or empty in inputdata.")

    # Select only the required columns.
    df = df[columns]

    # Identify columns that are not completely empty.
    non_empty_columns = [col for col in columns if not df[col].isna().all()]

    # Only drop rows with missing values in columns that have some data.
    if non_empty_columns:
        df = df.dropna(subset=non_empty_columns)
    else:
        warnings.warn(
            "All selected columns are empty. Returning the original dataframe slice."
        )

    return df


def _read_filtered_chunks(
    path: str, needed_columns: Set[str], inputdata: Inputdata
) -> Iterator[pd.DataFrame]:
    """
    Yields filtered DataFrame chunks from a CSV file.
    """
    for chunk in pd.read_csv(path, usecols=needed_columns, chunksize=10000):
        filtered_chunk = _apply_inputdata(chunk, inputdata)
        if not filtered_chunk.empty:
            yield filtered_chunk


def fetch_data(inputdata: Inputdata, csv_paths: List[str]) -> pd.DataFrame:
    """
    Loads CSV data from the given paths in chunks, applies filtering based on inputdata,
    and concatenates the results into a single DataFrame.

    Args:
        inputdata: An instance of Inputdata with filter and column selection instructions.
        csv_paths: A list of CSV file paths.

    Returns:
        A concatenated DataFrame after filtering.
    """
    x_columns = inputdata.x or []
    y_columns = inputdata.y or []
    # Include "dataset" column for filtering
    needed_columns = set(x_columns + y_columns + ["dataset"])

    # Gather filtered chunks from all CSV files
    chunks = []
    for path in csv_paths:
        chunks.extend(list(_read_filtered_chunks(path, needed_columns, inputdata)))

    return (
        pd.concat(chunks, ignore_index=True)
        if chunks
        else pd.DataFrame(columns=list(needed_columns))
    )
