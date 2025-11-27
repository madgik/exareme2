from exaflow.aggregation_clients.exaflow_udf_aggregation_client import (
    ExaflowUDFAggregationClient as AggregationClient,
)
from exaflow.algorithms.exaflow.exaflow_registry import exaflow_registry
from exaflow.algorithms.exaflow.longitudinal_transformer import (
    apply_longitudinal_transformation,
)
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.algorithms.utils.pandas_utils import ensure_pandas_dataframe
from exaflow.worker import config as worker_config
from exaflow.worker.exaflow.udf.udf_db import load_algorithm_arrow_table
from exaflow.worker.utils.logger import get_logger
from exaflow.worker.utils.logger import initialise_logger


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


@initialise_logger
def run_udf(
    request_id,
    udf_registry_key: str,
    params: dict,
):
    # TODO this has to be completely replace on algorithms to expect x and y not data
    transformed_inputdata_dict = params["inputdata"]
    transformed_inputdata = Inputdata.parse_raw(transformed_inputdata_dict)
    params["inputdata"] = transformed_inputdata
    loader_inputdata_dict = params.get("raw_inputdata", transformed_inputdata_dict)
    loader_inputdata = Inputdata.parse_raw(loader_inputdata_dict)
    params.pop("raw_inputdata", None)

    if exaflow_registry.aggregation_server_required(udf_registry_key):
        agg_dns = (
            getattr(getattr(worker_config, "aggregation_server", {}), "dns", None)
            or None
        )
        params["agg_client"] = AggregationClient(request_id, aggregator_dns=agg_dns)

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

    if preprocessing and "longitudinal_transformer" in preprocessing:
        data = apply_longitudinal_transformation(
            data, preprocessing["longitudinal_transformer"]
        )
    params["data"] = ensure_pandas_dataframe(data)
    udf = exaflow_registry.get_func(udf_registry_key)
    if not udf:
        error_msg = f"udf '{udf_registry_key}' not found in EXAFLOW_REGISTRY."
        raise ImportError(error_msg)

    try:
        result = udf(**params)
        return result
    except TypeError as e:
        logger.error(
            f"Error calling udf '{udf_registry_key}' with params {params}: {e}"
        )
        raise
