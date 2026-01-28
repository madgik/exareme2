from exaflow.aggregation_clients.exareme3_udf_aggregation_client import (
    Exareme3UDFAggregationClient as AggregationClient,
)
from exaflow.algorithms.exareme3.exareme3_registry import exareme3_registry
from exaflow.algorithms.exareme3.longitudinal_transformer import (
    apply_longitudinal_transformation,
)
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.algorithms.utils.pandas_utils import ensure_pandas_dataframe
from exaflow.worker import config as worker_config
from exaflow.worker.exareme3.udf.udf_db import load_algorithm_arrow_table
from exaflow.worker.utils.logger import get_logger
from exaflow.worker.utils.logger import initialise_logger
from exaflow.worker_communication import InsufficientDataError


def enforce_enum_order(data_dict):
    for key, field in data_dict.items():
        if field.get("enumerations"):
            ordered = field.get("ordered_enums")

            # Only process if ordered_enums exists
            if ordered and "enumerations" in field:
                enums = field["enumerations"]

                # Rebuild the enumerations dict using the list order
                new_enums = {code: enums[code] for code in ordered if code in enums}

                field["enumerations"] = new_enums

                # Remove the ordered_enums entry
                del field["ordered_enums"]
    return data_dict


def _coerce_series_to_enums(series, enums):
    import numpy as np
    import pandas as pd

    if not isinstance(series, pd.Series):
        return series
    enums_list = list(enums) if enums is not None else []
    if not enums_list:
        return series

    enums_are_ints = all(
        isinstance(e, (int, np.integer)) and not isinstance(e, bool) for e in enums_list
    )
    enums_are_strs = all(isinstance(e, str) for e in enums_list)

    if enums_are_ints:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() == series.notna().sum():
            numeric_non_null = numeric.dropna()
            if numeric_non_null.empty or np.all(
                np.isclose(numeric_non_null, np.round(numeric_non_null))
            ):
                return numeric.astype("Int64")
    elif enums_are_strs and not pd.api.types.is_string_dtype(series):
        return series.astype("string")

    return series


def coerce_categorical_columns(data, metadata):
    if data is None or metadata is None:
        return data
    for name, field in metadata.items():
        if not field.get("is_categorical"):
            continue
        enums = field.get("enumerations")
        if not enums or name not in data.columns:
            continue
        enum_values = enums.keys() if isinstance(enums, dict) else enums
        data[name] = _coerce_series_to_enums(data[name], enum_values)
    return data


@initialise_logger
def run_udf(
    request_id,
    udf_registry_key: str,
    params: dict,
):
    loader_inputdata_dict = params.get("raw_inputdata")
    loader_inputdata = Inputdata.parse_raw(loader_inputdata_dict)
    params.pop("raw_inputdata", None)

    if exareme3_registry.aggregation_server_required(udf_registry_key):
        agg_dns = (
            getattr(getattr(worker_config, "aggregation_server", {}), "dns", None)
            or None
        )
        agg_client = AggregationClient(request_id, aggregator_dns=agg_dns)
        params["agg_client"] = agg_client

    logger = get_logger()
    if "metadata" in params:
        # GRPC will mess with the order of dict when sending from controller to worker we need a list with the order to we can re-arrange them properly
        params["metadata"] = enforce_enum_order(params["metadata"])

    dropna = params.pop("dropna", True)
    include_dataset = params.pop("include_dataset", False)

    preprocessing = params.pop("preprocessing", None)
    extra_columns = set()
    if preprocessing and "longitudinal_transformer" in preprocessing:
        include_dataset = True
        extra_columns.update(
            preprocessing["longitudinal_transformer"].get("raw_x", [])
            + preprocessing["longitudinal_transformer"].get("raw_y", [])
        )
        extra_columns.update({"subjectid", "visitid"})

    data = load_algorithm_arrow_table(
        loader_inputdata,
        dropna=dropna,
        include_dataset=include_dataset,
        extra_columns=extra_columns if extra_columns else None,
    )
    num_rows = data.num_rows
    min_required = worker_config.privacy.minimum_row_count
    if num_rows < min_required:
        agg_client = params.get("agg_client")
        if agg_client:
            try:
                agg_client.unregister()
            finally:
                agg_client.close()
        raise InsufficientDataError(
            f"Insufficient data returned {num_rows} rows; minimum required is {min_required}."
        )

    if preprocessing and "longitudinal_transformer" in preprocessing:
        data = apply_longitudinal_transformation(
            data, preprocessing["longitudinal_transformer"]
        )
    params["data"] = ensure_pandas_dataframe(data)
    if "metadata" in params:
        params["data"] = coerce_categorical_columns(params["data"], params["metadata"])
    udf = exareme3_registry.get_func(udf_registry_key)
    if not udf:
        error_msg = f"udf '{udf_registry_key}' not found in EXAREME3_REGISTRY."
        raise ImportError(error_msg)

    if "metadata" in params:
        import inspect

        try:
            signature = inspect.signature(udf)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            parameters = signature.parameters
            accepts_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in parameters.values()
            )
            if "metadata" not in parameters and not accepts_kwargs:
                params.pop("metadata", None)

    try:
        result = udf(**params)
        return result
    except TypeError as e:
        logger.error(
            f"Error calling udf '{udf_registry_key}' with params {params}: {e}"
        )
        raise
