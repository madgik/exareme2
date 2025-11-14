from exareme2.aggregation_clients.exaflow_udf_aggregation_client import (
    ExaflowUDFAggregationClient as AggregationClient,
)
from exareme2.algorithms.exaflow.exaflow_registry import exaflow_registry
from exareme2.algorithms.utils.inputdata_utils import Inputdata
from exareme2.worker.utils.logger import get_logger
from exareme2.worker.utils.logger import initialise_logger
from exareme2.worker.worker_info.worker_info_db import get_dataset_csv_paths


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
