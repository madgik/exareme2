from exareme2.controller import logger as ctrl_logger
from exareme2.controller.services.tasks_handler_interface import TasksHandlerI
from exareme2.controller.worker_client.tasks_handler import WorkerTasksHandler


class FlowerTasksHandler(TasksHandlerI):
    def __init__(
        self,
        request_id: str,
        worker_id: str,
        worker_queue_addr: str,
        worker_db_addr: str,
        tasks_timeout: int,
    ):
        self._request_id = request_id
        self._worker_id = worker_id
        self._worker_queue_addr = worker_queue_addr
        self._db_address = worker_db_addr
        self._tasks_timeout = tasks_timeout
        self._logger = ctrl_logger.get_request_logger(request_id=request_id)
        self._worker_tasks_handler = WorkerTasksHandler(
            self._worker_queue_addr, self._logger
        )

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def worker_data_address(self) -> str:
        return self._db_address

    def start_flower_client(
        self,
        algorithm_folder_path,
        server_address,
        data_model,
        datasets,
        execution_timeout,
    ) -> int:
        return self._worker_tasks_handler.start_flower_client(
            request_id=self._request_id,
            algorithm_folder_path=algorithm_folder_path,
            server_address=server_address,
            data_model=data_model,
            datasets=datasets,
            execution_timeout=execution_timeout,
            timeout=self._tasks_timeout,
        )

    def start_flower_server(
        self,
        algorithm_folder_path: str,
        number_of_clients: int,
        server_address,
        data_model,
        datasets,
    ) -> int:
        return self._worker_tasks_handler.start_flower_server(
            request_id=self._request_id,
            algorithm_folder_path=algorithm_folder_path,
            number_of_clients=number_of_clients,
            server_address=server_address,
            data_model=data_model,
            datasets=datasets,
            timeout=self._tasks_timeout,
        )

    def stop_flower_server(self, pid: int, algorithm_name: str):
        self._worker_tasks_handler.stop_flower_server(
            request_id=self._request_id,
            pid=pid,
            algorithm_name=algorithm_name,
            timeout=self._tasks_timeout,
        )

    def stop_flower_client(self, pid: int, algorithm_name: str):
        self._worker_tasks_handler.stop_flower_client(
            request_id=self._request_id,
            pid=pid,
            algorithm_name=algorithm_name,
            timeout=self._tasks_timeout,
        )

    def garbage_collect(self):
        self._worker_tasks_handler.garbage_collect(
            request_id=self._request_id, timeout=self._tasks_timeout
        )
