from exareme2.algorithms.flower.process_manager import FlowerProcess
from exareme2.worker.utils.logger import get_logger
from exareme2.worker.utils.logger import initialise_logger


@initialise_logger
def stop_flower_process(request_id: str, pid: int, algorithm_name):
    logger = get_logger()
    FlowerProcess.kill_process(pid, algorithm_name, logger)
