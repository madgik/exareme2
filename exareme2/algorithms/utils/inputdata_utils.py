from typing import Any
from typing import Dict
from typing import List

import pandas as pd


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
    # Rename "id" to "column" to avoid shadowing built-in names.
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


def _apply_inputdata(df: pd.DataFrame, inputdata: Dict[str, Any]) -> pd.DataFrame:
    """
    Filters the dataframe based on the provided input data including filters, datasets, and selected columns.
    """
    if "filters" in inputdata and inputdata["filters"]:
        df = _apply_filter(df, inputdata["filters"])

    if "datasets" in inputdata:
        df = df[df["dataset"].isin(inputdata["datasets"])]
    else:
        raise ValueError("Missing 'datasets' key in inputdata.")

    x_columns = inputdata.get("x") or []
    y_columns = inputdata.get("y") or []
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
        # Optionally, you can warn or handle the case where all columns are empty.
        import warnings

        warnings.warn(
            "All selected columns are empty. Returning the original dataframe slice."
        )

    return df


def fetch_data(inputdata: Dict[str, Any], csv_paths: List[str]) -> pd.DataFrame:
    """
    Loads CSV data from the given paths in chunks, applies filtering based on inputdata,
    and concatenates the results into a single DataFrame.

    Args:
        inputdata: A dictionary with filter and column selection instructions.
        csv_paths: A list of CSV file paths.

    Returns:
        A concatenated DataFrame after filtering.
    """
    x_columns = inputdata.get("x") or []
    y_columns = inputdata.get("y") or []
    # Include "dataset" column for filtering
    needed_columns = set(x_columns + y_columns + ["dataset"])

    chunks = []
    for path in csv_paths:
        for chunk in pd.read_csv(path, usecols=needed_columns, chunksize=10000):
            # Apply filtering on the chunk if needed
            filtered_chunk = _apply_inputdata(chunk, inputdata)
            if not filtered_chunk.empty:
                chunks.append(filtered_chunk)
    if chunks:
        return pd.concat(chunks, ignore_index=True)
    else:
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=list(needed_columns))
