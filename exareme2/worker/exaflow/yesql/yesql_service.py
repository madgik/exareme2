import json
import time
from pathlib import Path

import yesql

from exareme2.algorithms.utils.inputdata_utils import Inputdata
from exareme2.worker import config as worker_config
from exareme2.worker.exareme2.views import views_db
from exareme2.worker.exareme2.views.views_service import (
    _get_filters_with_columns_not_null_constraints,
)
from exareme2.worker.exareme2.views.views_service import (
    _get_filters_with_datasets_constraints,
)
from exareme2.worker.exareme2.views.views_service import (
    _validate_data_model_and_datasets_exist,
)
from exareme2.worker.exareme2.views.views_service import create_data_model_views
from exareme2.worker.utils.logger import initialise_logger
from exareme2.worker.worker_info.worker_info_db import get_datasets

MINIMUM_ROW_COUNT = worker_config.privacy.minimum_row_count
MAX_EXECUTION_RETRIES = 5
WARNINGS_LOG_PATH = Path(__file__).with_name("warning.txt")


def _write_warning(message: str) -> None:
    with WARNINGS_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{message}\n")


@initialise_logger
def run_yesql(
    request_id,
    udf_registry_key: str,
    params: dict,
):
    inpudata_dict = params["inputdata"]
    inpudata = Inputdata.parse_raw(inpudata_dict)

    params["inputdata"] = inpudata
    datasets = inpudata.datasets
    data_model = inpudata.data_model
    datasets = get_datasets(data_model, datasets)
    filters = inpudata.filters

    _validate_data_model_and_datasets_exist(data_model, datasets)
    if datasets:
        filters = _get_filters_with_datasets_constraints(
            filters=filters, datasets=datasets
        )
    all_columns = (inpudata.x if inpudata.x else []) + (
        inpudata.y if inpudata.y else []
    )
    # In each view, it's not null constraints should include the columns of ALL views
    filters = _get_filters_with_columns_not_null_constraints(
        filters=filters, columns=all_columns
    )
    _view_name = f"{udf_registry_key}_{request_id}"
    views_db.create_view(
        view_name=_view_name,
        table_name=f'"{inpudata.data_model}"."primary_data"',
        columns=all_columns,
        filters=filters,
        minimum_row_count=MINIMUM_ROW_COUNT,
        check_min_rows=False,
    )
    start = time.perf_counter()

    times = 1
    connection = yesql.connect_init(
        worker_config.monetdb.local_username,
        worker_config.monetdb.local_password,
        worker_config.monetdb.ip,
        worker_config.monetdb.database,
        worker_config.monetdb.port,
        "monetdb",
        "/home/YeSQL/udfs",
    )

    for i in range(times):
        print(i)
        cursor = connection.cursor()
        result = None
        try:
            cursor.execute(
                f"cast 'string' run_on_exaflow '{udf_registry_key}' '{request_id}' select * from {_view_name};"
            )
            result = list(cursor.fetchall())
            _write_warning(f"{worker_config.monetdb.port=} {result=} ")
            _write_warning(f"{worker_config.monetdb.port=} {result[0][0]=} ")

        except Exception as exec_err:
            _write_warning(
                f" {worker_config.monetdb.port=} YeSQL execution failed: {exec_err}"
            )

    connection.close()

    elapsed = time.perf_counter() - start
    print(f"Total YeSQL time over {times} run(s): {elapsed}")
    path = Path("/home/kfilippopolitis/Desktop/metrics.txt")
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"Total YeSQL time over {times} run(s): {elapsed} on 200MB\n")

    return json.loads(result[0][0])
