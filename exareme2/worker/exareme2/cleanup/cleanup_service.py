from exareme2.worker.exareme2.cleanup.cleanup_db import drop_db_artifacts_by_context_id
from exareme2.worker.utils.logger import initialise_logger


@initialise_logger
def cleanup(request_id: str, context_id: str):
    """
    Parameters
    ----------
    request_id : str
        The identifier for the logging
    context_id : str
        The id of the experiment
    """
    drop_db_artifacts_by_context_id(context_id)
