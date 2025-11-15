from exareme2.algorithms.flower.process_manager import FlowerProcess
from exareme2.worker import config as worker_config
from exareme2.worker.utils.logger import get_logger
from exareme2.worker.utils.logger import initialise_logger
from exareme2.worker.worker_info.worker_info_db import get_dataset_csv_paths


@initialise_logger
def start_flower_client(
    request_id: str,
    algorithm_folder_path,
    server_address,
    data_model,
    datasets,
    execution_timeout,
) -> int:
    env_vars = {
        "REQUEST_ID": request_id,
        "FEDERATION": worker_config.federation,
        "WORKER_IDENTIFIER": worker_config.identifier,
        "WORKER_ROLE": worker_config.role,
        "LOG_LEVEL": worker_config.log_level,
        "FRAMEWORK_LOG_LEVEL": worker_config.framework_log_level,
        "SERVER_ADDRESS": server_address,
        "CONTROLLER_IP": worker_config.controller.ip,
        "CONTROLLER_PORT": worker_config.controller.port,
        "DATA_PATH": worker_config.data_path,
        "CSV_PATHS": ",".join(get_dataset_csv_paths(data_model, datasets)),
        "TIMEOUT": execution_timeout,
    }
    process = FlowerProcess(f"{algorithm_folder_path}/client.py", env_vars=env_vars)
    logger = get_logger()

    logger.info("Starting flower client...")
    pid = process.start(logger)
    logger.info(f"Started flower client, with process id: {pid}")
    return pid


@initialise_logger
def start_flower_server(
    request_id: str,
    algorithm_folder_path: str,
    number_of_clients: int,
    server_address,
    data_model,
    datasets,
) -> int:
    env_vars = {
        "FEDERATION": worker_config.federation,
        "WORKER_ROLE": worker_config.role,
        "WORKER_IDENTIFIER": worker_config.identifier,
        "LOG_LEVEL": worker_config.log_level,
        "FRAMEWORK_LOG_LEVEL": worker_config.framework_log_level,
        "REQUEST_ID": request_id,
        "SERVER_ADDRESS": server_address,
        "NUMBER_OF_CLIENTS": number_of_clients,
        "CONTROLLER_IP": worker_config.controller.ip,
        "CONTROLLER_PORT": worker_config.controller.port,
        "DATA_PATH": worker_config.data_path,
        "CSV_PATHS": ",".join(get_dataset_csv_paths(data_model, datasets)),
    }

    process = FlowerProcess(f"{algorithm_folder_path}/server.py", env_vars=env_vars)
    logger = get_logger()
    logger.info("Starting flower server...")
    pid = process.start(logger)
    logger.info(f"Started flower server, with process id: {pid}")
    return pid
