from typing import Any
from typing import Dict
from typing import List
from typing import Optional

#  ~~~ ATTENTION ~~~ Changing these logs will have implications on the federation_info script.


def log_experiment_execution(
    logger,
    request_id: str,
    context_id: str,
    algorithm_name: str,
    datasets: List[str],
    algorithm_parameters: Optional[Dict[str, Any]],
    local_worker_ids: List[str],
):
    logger.info(
        f"Experiment with request id '{request_id}' and context id '{context_id}' is starting "
        f"algorithm '{algorithm_name}', touching datasets '{','.join(datasets)}' on local "
        f"workers '{','.join(local_worker_ids)}' with parameters '{algorithm_parameters}'."
    )


def log_worker_left_federation(logger, worker):
    logger.info(f"Worker with id '{worker}' left the federation.")


def log_worker_joined_federation(logger, worker):
    logger.info(f"Worker with id '{worker}' joined the federation.")


def log_datamodel_removed(data_model, logger):
    logger.info(f"Datamodel '{data_model}' was removed.")


def log_datamodel_added(data_model, logger):
    logger.info(f"Datamodel '{data_model}' was added.")


def log_dataset_added(data_model, dataset, logger, worker_id):
    logger.info(
        f"Dataset '{dataset}' of datamodel '{data_model}' was added in worker '{worker_id}'."
    )


def log_dataset_removed(data_model, dataset, logger, worker_id):
    logger.info(
        f"Dataset '{dataset}' of datamodel '{data_model}' was removed from worker '{worker_id}'."
    )
