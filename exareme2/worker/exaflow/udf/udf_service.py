import inspect
import textwrap

from exareme2.algorithms.exaflow.exaflow_registry import exaflow_registry
from exareme2.algorithms.utils.inputdata_utils import Inputdata
from exareme2.worker.exareme2.monetdb import monetdb_facade
from exareme2.worker.utils.logger import get_logger
from exareme2.worker.utils.logger import initialise_logger
from exareme2.worker.worker_info.worker_info_db import get_dataset_csv_paths


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
    inpudata_dict = params["inputdata"]
    inpudata = Inputdata.parse_raw(inpudata_dict)
    params["inputdata"] = inpudata
    params["csv_paths"] = get_dataset_csv_paths(inpudata.data_model, inpudata.datasets)

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


def create_udf(request_id: str, func) -> None:
    """
    Constructs and executes the CREATE OR REPLACE FUNCTION SQL statement for the given UDF.

    This version:
      - Extracts and dedents the source code of 'func'
      - Replaces the full import ("from exareme2.aggregator.aggregator_client import AggregationClient")
        with the local import ("from aggregator_client import AggregationClient")
      - Replaces the request id placeholder "<here goes the request_id>" with the actual request id
      - Removes only the function definition line (i.e. any line starting with "def ")
        so that any preceding decorators (e.g. @exaflow_udf) remain in the UDF code.
      - Reindents the resulting code for correct insertion into the SQL block.

    Finally, it builds and executes the CREATE FUNCTION SQL via monetdb_facade.
    """
    # Retrieve and dedent the source code.
    source_code = inspect.getsource(func)
    source_code = textwrap.dedent(source_code)

    # Replace the full import with the local import.
    source_code = source_code.replace(
        "from exareme2.aggregator.aggregator_client import AggregationClient",
        "from aggregator_client import AggregationClient",
    ).replace("@exaflow_udf", "")

    # Replace the request id placeholder with the actual request id.
    source_code = source_code.replace("<here goes the request_id>", f"{request_id}")

    # Split the source into lines.
    lines = source_code.splitlines()
    # Filter out any line that starts with "def " (i.e. the function definition header) so that decorators are preserved.
    filtered_lines = [line for line in lines if not line.lstrip().startswith("def ")]

    # Rejoin and dedent the remaining lines, then indent them for SQL formatting.
    udf_body = "\n".join(filtered_lines)
    udf_body = textwrap.dedent(udf_body)
    udf_body = textwrap.indent(udf_body, "    ")

    # Use the function's original name as the UDF name.
    udf_name = func.__name__

    # Build the CREATE OR REPLACE FUNCTION SQL statement.
    create_sql = f"""
CREATE OR REPLACE FUNCTION {udf_name}(data FLOAT) RETURNS FLOAT LANGUAGE PYTHON {{
{udf_body}
}};
"""
    create_sql
    # Execute the SQL statement via monetdb_facade.
    monetdb_facade.execute_query(create_sql)


def execute_udf(inputdata: dict, udf_name: str):
    """
    Constructs and executes a SELECT query that invokes the deployed UDF.

    inputdata must contain:
      - "data_model": Schema name (e.g., "dementia:0.1")
      - "datasets": List of dataset names (used for filtering in the WHERE clause)
      - "y": List of column names (we use the first one)

    Returns:
      The result of executing the SELECT query.
    """
    data_model = inputdata["data_model"]
    datasets = inputdata["datasets"]
    y_col = inputdata["y"][0]

    # Format datasets list as comma-separated quoted strings.
    datasets_str = ", ".join([f"'{d}'" for d in datasets])
    udf_name = "local_step"
    select_sql = f"""
SELECT {udf_name}({y_col})
FROM "{data_model}"."primary_data"
WHERE dataset IN ({datasets_str});
"""
    return monetdb_facade.execute_and_fetchall(select_sql)


@initialise_logger
def run_monetdb_udf(request_id, udf_name: str, params: dict):
    """
    Executes a registered exaflow algorithm step:
      1. Obtains the input data and additional parameters.
      2. Retrieves the registered UDF from monetdb_udf_registry.
      3. Calls create_udf to deploy the UDF in MonetDB.
      4. Calls execute_udf to run the UDF on the target table/column.

    Returns:
      The result of the SELECT query.
    """
    inpudata_dict = params["inputdata"]
    inpudata = Inputdata.parse_raw(inpudata_dict)
    params["inputdata"] = inpudata
    params["csv_paths"] = get_dataset_csv_paths(inpudata.data_model, inpudata.datasets)

    # Retrieve the function from the registry.
    udf = exaflow_registry.get_udf(udf_name)

    logger = get_logger()
    if not udf:
        error_msg = f"udf '{udf_name}' not found in EXAFLOW_REGISTRY."
        raise ImportError(error_msg)

    try:
        # Deploy the UDF in MonetDB.
        create_udf(request_id, udf)
        # Execute the UDF and return results.
        return execute_udf(inpudata.dict(), udf_name)[0][0]
    except TypeError as e:
        logger.error(f"Error calling udf '{udf_name}' with params {params}: {e}")
        raise
