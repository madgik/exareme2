from exareme2 import ALGORITHM_FOLDERS
from exareme2.algorithms.flower.process_manager import FlowerProcess
from exareme2.worker import config as worker_config
from exareme2.worker.utils.logger import get_logger
from exareme2.worker.utils.logger import initialise_logger

# Dictionary to keep track of running processes
running_processes = {}
SERVER_ADDRESS = "0.0.0.0:8080"


@initialise_logger
def start_flower_client(request_id: str, algorithm_name, worker_id) -> int:
    env_vars = {
        "MONETDB_IP": worker_config.monetdb.ip,
        "MONETDB_PORT": worker_config.monetdb.port,
        "MONETDB_USERNAME": worker_config.monetdb.local_username,
        "MONETDB_PASSWORD": worker_config.monetdb.local_password,
        "MONETDB_DB": worker_config.monetdb.database,
        "SERVER_ADDRESS": SERVER_ADDRESS,
        "NUMBER_OF_CLIENTS": worker_config.monetdb.database,
    }
    with open(f"/tmp/exareme2/{worker_id}.out", "a") as f:
        process = FlowerProcess(
            f"{algorithm_name}/client.py", env_vars=env_vars, stderr=f, stdout=f
        )
        running_processes[request_id] = process
        logger = get_logger()

        logger.info("Starting client.py")
        pid = process.start(logger)
    logger.info(f"Started client.py process id: {pid}")
    return pid


@initialise_logger
def start_flower_server(
    request_id: str, algorithm_name: str, number_of_clients: int, worker_id
) -> int:
    env_vars = {
        "SERVER_ADDRESS": SERVER_ADDRESS,
        "NUMBER_OF_CLIENTS": number_of_clients,
    }
    with open(f"/tmp/exareme2/{worker_id}.out", "a") as f:
        process = FlowerProcess(
            f"{algorithm_name}/server.py", env_vars=env_vars, stderr=f, stdout=f
        )
        running_processes[request_id] = process
        logger = get_logger()
        logger.info("Starting server.py")
        pid = process.start(logger)
    logger.info(f"Started server.py process id: {pid}")
    return pid
