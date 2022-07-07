from typing import List

#  ~~~ ATTENTION ~~~ Changing these logs will have implications on the federation_info script.


def log_experiment_execution(
    logger,
    request_id: str,
    algorithm_name: str,
    datasets: List[str],
    algorithm_parameters: str,
):
    logger.info(
        f"Experiment with request id '{request_id}' started, "
        f"with algorithm '{algorithm_name}', "
        f"touching datasets '{','.join(datasets)}', "
        f"with parameters '{algorithm_parameters}'."
    )


def log_node_left_federation(logger, node):
    logger.info(f"Node with id '{node}' left the federation.")


def log_node_joined_federation(logger, node):
    logger.info(f"Node with id '{node}' joined the federation.")


def log_datamodel_removed(data_model, logger):
    logger.info(f"Datamodel '{data_model}' was removed.")


def log_datamodel_added(data_model, logger):
    logger.info(f"Datamodel '{data_model}' was added.")


def log_dataset_added(data_model, dataset, logger, node_id):
    logger.info(
        f"Dataset '{dataset}' of datamodel '{data_model}' was added in node '{node_id}'."
    )


def log_dataset_removed(data_model, dataset, logger, node_id):
    logger.info(
        f"Dataset '{dataset}' of datamodel '{data_model}' was removed from node '{node_id}'."
    )


def log_dataset_dropped(data_model, dataset, logger, node_id):
    logger.info(
        f"Dataset '{dataset}' of datamodel '{data_model}' was dropped from node '{node_id}'."
    )
