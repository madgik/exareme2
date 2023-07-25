import json

from exareme2.udfgen import secure_transfer
from exareme2.udfgen import transfer
from exareme2.udfgen import udf


def get_transfer_data(udf_result) -> dict:
    """
    Extracts data from transfer object stored in a table resulting from a UDF

    Parameters
    ----------
    udf_result : GlobalNodeData
        Result of run_udf_on_global_node when the corresponding UDF returns a transfer
    """
    [[val]] = udf_result.get_table_data()  # get_table_data returns List[List[str]]
    return json.loads(val)


@udf(loctransf=secure_transfer(sum_op=True), return_type=transfer())
def sum_secure_transfers(loctransf):
    return loctransf
