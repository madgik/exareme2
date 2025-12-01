import inspect
import threading
import time
import tracemalloc
from typing import Optional

import psutil

from exaflow.aggregation_clients.exaflow_udf_aggregation_client import (
    ExaflowUDFAggregationClient as AggregationClient,
)
from exaflow.algorithms.exaflow.exaflow_registry import exaflow_registry
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.worker import config as worker_config
from exaflow.worker.exaflow.udf.cursor import DuckDBCursor
from exaflow.worker.utils.logger import get_logger
from exaflow.worker.utils.logger import initialise_logger

BYTES_IN_MB = 1024 * 1024


class MemoryUsageTracker:
    def __init__(self, logger, udf_key: str, interval: float = 0.1):
        self._logger = logger
        self._udf_key = udf_key
        self._process = psutil.Process()
        self._start_rss: Optional[int] = None
        self._peak_rss: Optional[int] = None
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self):
        self._start_rss = self._current_rss()
        self._peak_rss = self._start_rss
        tracemalloc.start()
        self._logger.info(
            "[MEMORY] %s start RSS %.2f MB",
            self._udf_key,
            self._start_rss / BYTES_IN_MB,
        )
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        peak_rss = self._peak_rss or self._current_rss()
        end_rss = self._current_rss()
        start_rss = self._start_rss or end_rss
        delta_mb = (end_rss - start_rss) / BYTES_IN_MB
        peak_mb = peak_rss / BYTES_IN_MB
        current_alloc = peak_alloc = None
        if tracemalloc.is_tracing():
            current_alloc, peak_alloc = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        current_mb = (current_alloc or 0) / BYTES_IN_MB
        peak_python_mb = (peak_alloc or 0) / BYTES_IN_MB
        self._logger.info(
            "[MEMORY] %s end RSS %.2f MB (Δ %.2f MB, peak %.2f MB) | Python alloc %.2f MB (peak %.2f MB)",
            self._udf_key,
            end_rss / BYTES_IN_MB,
            delta_mb,
            peak_mb,
            current_mb,
            peak_python_mb,
        )

    def _current_rss(self) -> int:
        return self._process.memory_info().rss

    def _sample_loop(self):
        while not self._stop_event.is_set():
            self._update_peak()
            time.sleep(self._interval)
        self._update_peak()

    def _update_peak(self):
        current = self._current_rss()
        if self._peak_rss is None or current > self._peak_rss:
            self._peak_rss = current


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

    cursor = DuckDBCursor(
        loader_inputdata,
        dropna=dropna,
        include_dataset=include_dataset,
        extra_columns=extra_columns if extra_columns else None,
        preprocessing=preprocessing,
    )
    udf = exaflow_registry.get_func(udf_registry_key)
    if not udf:
        error_msg = f"udf '{udf_registry_key}' not found in EXAFLOW_REGISTRY."
        raise ImportError(error_msg)

    param_names = set(inspect.signature(udf).parameters)
    if "cursor" in param_names:
        params["cursor"] = cursor
    if "data" in param_names:
        params["data"] = cursor.load_all_dataframe()

    try:
        with MemoryUsageTracker(logger, udf_registry_key):
            result = udf(**params)
        return result
    except TypeError as e:
        logger.error(
            f"Error calling udf '{udf_registry_key}' with params {params}: {e}"
        )
        raise
