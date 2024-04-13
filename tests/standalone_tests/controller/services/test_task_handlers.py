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

    def test_get_table_data(self):
        table_name = "test_table"
        self.mock_celery_app.queue_task.return_value = self.mock_async_result
        result = self.worker_tasks_handler.get_table_data(self.request_id, table_name)

        self.assertIsInstance(result, WorkerTaskResult)
        self.mock_celery_app.queue_task.assert_called_with(
            task_signature="exareme2.worker.exareme2.tables.tables_api.get_table_data",
            logger=self.mock_logger,
            request_id=self.request_id,
            table_name=table_name,
        )

    def test_create_table(self):
        schema = MagicMock()  # Assuming TableSchema instances can be mocked
        self.mock_celery_app.queue_task.return_value = self.mock_async_result
        result = self.worker_tasks_handler.create_table(
            self.request_id, self.context_id, self.command_id, schema
        )

        self.assertIsInstance(result, WorkerTaskResult)
        schema.json.assert_called_once()
        self.mock_celery_app.queue_task.assert_called_with(
            task_signature="exareme2.worker.exareme2.tables.tables_api.create_table",
            logger=self.mock_logger,
            request_id=self.request_id,
            context_id=self.context_id,
            command_id=self.command_id,
            schema_json=schema.json(),
        )

    def test_get_views(self):
        self.mock_celery_app.queue_task.return_value = self.mock_async_result
        result = self.worker_tasks_handler.get_views(self.request_id, self.context_id)

        self.assertIsInstance(result, WorkerTaskResult)
        self.mock_celery_app.queue_task.assert_called_with(
            task_signature="exareme2.worker.exareme2.views.views_api.get_views",
            logger=self.mock_logger,
            request_id=self.request_id,
            context_id=self.context_id,
        )

    def test_create_data_model_views(self):
        data_model = "test_data_model"
        datasets = ["dataset1", "dataset2"]
        columns_per_view = [["col1", "col2"], ["col3"]]
        filters = {"filter1": "value1"}
        dropna = True
        check_min_rows = True
        self.mock_celery_app.queue_task.return_value = self.mock_async_result
        result = self.worker_tasks_handler.create_data_model_views(
            self.request_id,
            self.context_id,
            self.command_id,
            data_model,
            datasets,
            columns_per_view,
            filters,
            dropna,
            check_min_rows,
        )

        self.assertIsInstance(result, WorkerTaskResult)
        self.mock_celery_app.queue_task.assert_called_with(
            task_signature="exareme2.worker.exareme2.views.views_api.create_data_model_views",
            logger=self.mock_logger,
            request_id=self.request_id,
            context_id=self.context_id,
            command_id=self.command_id,
            data_model=data_model,
            datasets=datasets,
            columns_per_view=columns_per_view,
            filters=filters,
            dropna=dropna,
            check_min_rows=check_min_rows,
        )

    def test_get_merge_tables(self):
        self.mock_celery_app.queue_task.return_value = self.mock_async_result
        result = self.worker_tasks_handler.get_merge_tables(
            self.request_id, self.context_id
        )

        self.assertIsInstance(result, WorkerTaskResult)
        self.mock_celery_app.queue_task.assert_called_with(
            task_signature="exareme2.worker.exareme2.tables.tables_api.get_merge_tables",
            logger=self.mock_logger,
            request_id=self.request_id,
            context_id=self.context_id,
        )

    def test_create_merge_table(self):
        table_infos = [
            MagicMock(),
            MagicMock(),
        ]  # Assuming TableInfo instances can be mocked
        self.mock_celery_app.queue_task.return_value = self.mock_async_result
        result = self.worker_tasks_handler.create_merge_table(
            self.request_id, self.context_id, self.command_id, table_infos
        )

        self.assertIsInstance(result, WorkerTaskResult)
        expected_table_infos_json = [info.json() for info in table_infos]
        self.mock_celery_app.queue_task.assert_called_with(
            task_signature="exareme2.worker.exareme2.tables.tables_api.create_merge_table",
            logger=self.mock_logger,
            request_id=self.request_id,
            context_id=self.context_id,
            command_id=self.command_id,
            table_infos_json=expected_table_infos_json,
        )

    def test_get_remote_tables(self):
        self.mock_celery_app.queue_task.return_value = self.mock_async_result
        result = self.worker_tasks_handler.get_remote_tables(
            self.request_id, self.context_id
        )

        self.assertIsInstance(result, WorkerTaskResult)
        self.mock_celery_app.queue_task.assert_called_with(
            task_signature="exareme2.worker.exareme2.tables.tables_api.get_remote_tables",
            logger=self.mock_logger,
            request_id=self.request_id,
            context_id=self.context_id,
        )

    def test_create_remote_table(self):
        table_name = "remote_table"
        table_schema = MagicMock()  # Assuming TableSchema instances can be mocked
        monetdb_socket_address = "fake_socket_address"
        self.mock_celery_app.queue_task.return_value = self.mock_async_result
        result = self.worker_tasks_handler.create_remote_table(
            self.request_id, table_name, table_schema, monetdb_socket_address
        )

        self.assertIsInstance(result, WorkerTaskResult)
        table_schema.json.assert_called_once()
        self.mock_celery_app.queue_task.assert_called_with(
            task_signature="exareme2.worker.exareme2.tables.tables_api.create_remote_table",
            logger=self.mock_logger,
            table_name=table_name,
            table_schema_json=table_schema.json(),
            monetdb_socket_address=monetdb_socket_address,
            request_id=self.request_id,
        )
