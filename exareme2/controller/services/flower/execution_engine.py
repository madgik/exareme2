from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional

from exareme2.controller import logger as ctrl_logger
from exareme2.controller.services.flower.workers import GlobalWorker
from exareme2.controller.services.flower.workers import LocalWorker


@dataclass
class Workers:
    local_workers: List[LocalWorker]
    global_worker: Optional[GlobalWorker] = None


class AlgorithmExecutionEngine:
    """
    Manages distributed algorithm execution using the Flower framework,
    enforcing specific configurations of workers.
    """

    def __init__(self, request_id: str, workers: Workers):
        self.logger = ctrl_logger.get_request_logger(request_id=request_id)
        self.workers = workers

    def start_flower(self, algorithm_name: str) -> Dict[str, Dict[str, int]]:
        """
        Initializes Flower servers and clients for a given algorithm.
        """
        return self.start_workers(algorithm_name)

    def stop_flower(self, process_ids: Dict[str, Dict[str, int]], algorithm_name: str):
        """
        Shuts down Flower servers and clients using their process IDs.
        """
        for worker_id, ids in process_ids.items():
            self.stop_worker(worker_id, ids, algorithm_name)

    def start_workers(self, algorithm_name: str) -> Dict[str, Dict[str, int]]:
        """
        Manages the initialization of Flower framework across multiple workers including a global worker.
        """
        process_ids = {
            self.workers.global_worker.worker_id: {
                "server": self.workers.global_worker.start_flower_server(
                    algorithm_name, len(self.workers.local_workers)
                )
            }
        }

        for worker in self.workers.local_workers:
            process_ids[worker.worker_id] = {
                "client": worker.start_flower_client(algorithm_name)
            }
        return process_ids

    def stop_worker(self, worker_id: str, ids: Dict[str, int], algorithm_name: str):
        """
        Stops the server or client processes on a specified worker.
        """
        worker = self.get_worker_by_id(worker_id)
        if worker:
            if "server" in ids:
                worker.stop_flower_server(ids["server"], algorithm_name)
            if "client" in ids:
                worker.stop_flower_client(ids["client"], algorithm_name)

    def get_worker_by_id(self, worker_id: str) -> Optional[LocalWorker]:
        """
        Retrieves a worker object by its ID.
        """
        if (
            self.workers.global_worker
            and self.workers.global_worker.worker_id == worker_id
        ):
            return self.workers.global_worker
        return next(
            (
                worker
                for worker in self.workers.local_workers
                if worker.worker_id == worker_id
            ),
            None,
        )

    @property
    def request_id(self) -> str:
        return self.workers.local_workers[0].request_id

    @property
    def context_id(self) -> str:
        return self.workers.local_workers[0].context_id
