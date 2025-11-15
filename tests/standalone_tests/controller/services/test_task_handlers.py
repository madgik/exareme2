import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from exareme2.controller import logger as ctrl_logger
from exareme2.controller.celery.app import CeleryAppFactory
from exareme2.controller.celery.tasks_handler import WorkerTaskResult
from exareme2.controller.celery.tasks_handler import WorkerTasksHandler


class TestWorkerTasksHandlerRefactored(unittest.TestCase):
    def setUp(self):
        self.mock_celery_app = MagicMock()
        self.mock_async_result = MagicMock()
        self.request_id = "test_request_id"
        self.context_id = "test_context_id"
        self.command_id = "test_command_id"
        self.mock_logger = ctrl_logger.get_request_logger(request_id=self.request_id)

        self.patcher_get_celery_app = patch.object(
            CeleryAppFactory, "get_celery_app", return_value=self.mock_celery_app
        )
        self.mock_get_celery_app = self.patcher_get_celery_app.start()

        self.worker_tasks_handler = WorkerTasksHandler(
            worker_queue_addr="fake_addr", logger=self.mock_logger
        )

    def tearDown(self):
        self.patcher_get_celery_app.stop()
