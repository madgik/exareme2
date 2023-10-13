from exareme2.node.logger import initialise_logger
from exareme2.node.monetdb.cleanup import drop_db_artifacts_by_context_id


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
