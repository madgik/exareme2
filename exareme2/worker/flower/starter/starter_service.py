from exareme2.algorithms.flower.process_manager import FlowerProcess
from exareme2.worker import config as worker_config
from exareme2.worker.utils.logger import get_logger
from exareme2.worker.utils.logger import initialise_logger


@initialise_logger
def start_flower_client(
    request_id: str, algorithm_folder_path, server_address, csv_paths, execution_timeout
) -> int:
    env_vars = {
        "MONETDB_IP": worker_config.monetdb.ip,
        "MONETDB_PORT": worker_config.monetdb.port,
        "MONETDB_USERNAME": worker_config.monetdb.local_username,
        "MONETDB_PASSWORD": worker_config.monetdb.local_password,
        "MONETDB_DB": worker_config.monetdb.database,
        "REQUEST_ID": request_id,
        "WORKER_ROLE": worker_config.role,
        "WORKER_IDENTIFIER": worker_config.identifier,
        "SERVER_ADDRESS": server_address,
        "NUMBER_OF_CLIENTS": worker_config.monetdb.database,
        "CONTROLLER_IP": worker_config.controller.ip,
        "CONTROLLER_PORT": worker_config.controller.port,
        "DATA_PATH": worker_config.data_path,
        "CSV_PATHS": ",".join(csv_paths),
        "TIMEOUT": execution_timeout,
    }
    process = FlowerProcess(f"{algorithm_folder_path}/client.py", env_vars=env_vars)
    logger = get_logger()

    logger.info("Starting client.py")
    pid = process.start(logger)
    logger.info(f"Started client.py process id: {pid}")
    return pid


@initialise_logger
def start_flower_server(
    request_id: str,
    algorithm_folder_path: str,
    number_of_clients: int,
    server_address,
    csv_paths,
) -> int:
    env_vars = {
        "REQUEST_ID": request_id,
        "WORKER_ROLE": worker_config.role,
        "WORKER_IDENTIFIER": worker_config.identifier,
        "SERVER_ADDRESS": server_address,
        "NUMBER_OF_CLIENTS": number_of_clients,
        "CONTROLLER_IP": worker_config.controller.ip,
        "CONTROLLER_PORT": worker_config.controller.port,
        "DATA_PATH": worker_config.data_path,
        "CSV_PATHS": ",".join(csv_paths),
    }
    process = FlowerProcess(f"{algorithm_folder_path}/server.py", env_vars=env_vars)
    logger = get_logger()
    logger.info("Starting server.py")
    pid = process.start(logger)
    logger.info(f"Started server.py process id: {pid}")
    return pid
