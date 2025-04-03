from exareme2.algorithms.exaflow.exaflow_registry import exaflow_registry
from exareme2.worker import config as worker_config
from exareme2.worker.utils.logger import get_logger
from exareme2.worker.utils.logger import initialise_logger


@initialise_logger
def run_udf(
    request_id,
    udf_name: str,
    params: dict,
):
    """
    Executes a registered exaflow algorithm step.

    Args:
        request_id (str): Unique identifier for the request.
        udf_name (str): Name of the registered udf to execute (e.g., 'local_step').
        params (dict): Parameters required by the udf.

    Returns:
        Any: Result returned by the algorithm step.
    """
    params["csv_paths"] = [
        f"{worker_config.data_path}/{csv_path}" for csv_path in params["csv_paths"]
    ]
    udf = exaflow_registry.get_udf(udf_name)
    logger = get_logger()

    if not udf:
        error_msg = f"udf '{udf_name}' not found in EXAFLOW_REGISTRY."
        raise ImportError(error_msg)

    try:
        result = udf(**params)
        return result
    except TypeError as e:
        logger.error(f"Error calling udf '{udf_name}' with params {params}: {e}")
        raise
