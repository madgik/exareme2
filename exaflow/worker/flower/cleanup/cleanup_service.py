from exaflow.algorithms.flower.process_manager import FlowerProcess
from exaflow.worker.utils.logger import get_logger
from exaflow.worker.utils.logger import initialise_logger


@initialise_logger
def stop_flower_process(request_id: str, pid: int, algorithm_name):
    logger = get_logger()
    FlowerProcess.kill_process(pid, algorithm_name, logger)


@initialise_logger
def garbage_collect(request_id: str):
    logger = get_logger()
    FlowerProcess.garbage_collect(logger)
