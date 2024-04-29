from typing import List

from exareme2.controller.services.flower.tasks_handler import FlowerTasksHandler


class Worker:
    def __init__(
        self, request_id: str, context_id: str, worker_tasks_handler: FlowerTasksHandler
    ):
        self._worker_tasks_handler = worker_tasks_handler
        self.worker_id = self._worker_tasks_handler.worker_id
        self.request_id = request_id
        self.context_id = context_id

    def __repr__(self):
        return f"{self.worker_id}"

    @property
    def worker_address(self) -> str:
        return self._worker_tasks_handler.worker_data_address

    def start_flower_server(self, algorithm_name, number_of_clients) -> int:
        return self._worker_tasks_handler.start_flower_server(
            algorithm_name, number_of_clients
        )

    def stop_flower_server(self, pid: int, algorithm_name: str):
        self._worker_tasks_handler.stop_flower_server(pid, algorithm_name)

    def start_flower_client(self, algorithm_name) -> int:
        return self._worker_tasks_handler.start_flower_client(algorithm_name)

    def stop_flower_client(self, pid: int, algorithm_name: str):
        self._worker_tasks_handler.stop_flower_client(pid, algorithm_name)


class LocalWorker(Worker):
    def __init__(
        self,
        request_id: str,
        context_id: str,
        flower_tasks_handler: FlowerTasksHandler,
        data_model: str,
        datasets: List[str],
    ):
        super().__init__(request_id, context_id, flower_tasks_handler)
        self._data_model = data_model
        self._datasets = datasets

    @property
    def data_model(self):
        return self._data_model

    @property
    def datasets(self):
        return self._datasets


class GlobalWorker(Worker):
    pass  # GlobalWorker has no additional methods over what is defined in Worker
