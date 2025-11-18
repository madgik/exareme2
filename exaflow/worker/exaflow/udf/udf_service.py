from exaflow.aggregation_clients.exaflow_udf_aggregation_client import (
    ExaflowUDFAggregationClient as AggregationClient,
)
from exaflow.algorithms.exaflow.exaflow_registry import exaflow_registry
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.worker.utils.logger import get_logger
from exaflow.worker.utils.logger import initialise_logger
from exaflow.worker.worker_info.worker_info_db import get_dataset_csv_paths


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
    inpudata_dict = params["inputdata"]
    inpudata = Inputdata.parse_raw(inpudata_dict)
    params["inputdata"] = inpudata
    params["csv_paths"] = get_dataset_csv_paths(inpudata.data_model, inpudata.datasets)

    if exaflow_registry.aggregation_server_required(udf_registry_key):
        params["agg_client"] = AggregationClient(request_id)

    logger = get_logger()
    if "metadata" in params:
        # GRPC will mess with the order of dict when sending from controller to worker we need a list with the order to we can re-arrange them properly
        params["metadata"] = enforce_enum_order(params["metadata"])
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
