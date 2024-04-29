import unittest
from unittest.mock import MagicMock
from unittest.mock import call
from unittest.mock import patch

from exareme2.controller.services.flower.execution_engine import (
    AlgorithmExecutionEngine,
)
from exareme2.controller.services.flower.execution_engine import Workers
from exareme2.controller.services.flower.workers import GlobalWorker
from exareme2.controller.services.flower.workers import LocalWorker


class TestAlgorithmExecutionEngine(unittest.TestCase):
    @patch("exareme2.controller.logger.get_request_logger")
    def test_multiple_local_workers_with_global(self, mock_logger):
        # Multiple local workers and a global worker
        local_worker1 = MagicMock(spec=LocalWorker, worker_id="local_worker1")
        local_worker2 = MagicMock(spec=LocalWorker, worker_id="local_worker2")
        global_worker = MagicMock(spec=GlobalWorker, worker_id="global_worker")
        workers = Workers(
            local_workers=[local_worker1, local_worker2], global_worker=global_worker
        )
        engine = AlgorithmExecutionEngine("req123", workers)

    def test_start_stop_flower_single_local(self):
        local_worker = MagicMock(spec=LocalWorker, worker_id="local_worker")
        workers = Workers(local_workers=[local_worker])
        engine = AlgorithmExecutionEngine("req123", workers)

        with patch.object(engine, "start_flower") as mock_start, patch.object(
            engine, "stop_flower"
        ) as mock_stop:
            engine.start_flower("alg1")
            engine.stop_flower({"local_worker": {"server": 1, "client": 2}}, "alg1")

            mock_start.assert_called_once_with("alg1")
            mock_stop.assert_called_once_with(
                {"local_worker": {"server": 1, "client": 2}}, "alg1"
            )

    def test_start_stop_flower_multiple_local_with_global(self):
        local_worker1 = MagicMock(spec=LocalWorker, worker_id="local_worker1")
        local_worker2 = MagicMock(spec=LocalWorker, worker_id="local_worker2")
        global_worker = MagicMock(spec=GlobalWorker, worker_id="global_worker")
        workers = Workers(
            local_workers=[local_worker1, local_worker2], global_worker=global_worker
        )
        engine = AlgorithmExecutionEngine("req123", workers)

        with patch.object(engine, "start_flower") as mock_start, patch.object(
            engine, "stop_flower"
        ) as mock_stop:
            engine.start_flower("alg1")
            engine.stop_flower(
                {
                    "global_worker": {"server": 1},
                    "local_worker1": {"client": 2},
                    "local_worker2": {"client": 3},
                },
                "alg1",
            )

            mock_start.assert_called_once_with("alg1")
            mock_stop.assert_called_once_with(
                {
                    "global_worker": {"server": 1},
                    "local_worker1": {"client": 2},
                    "local_worker2": {"client": 3},
                },
                "alg1",
            )

    @patch("exareme2.controller.logger.get_request_logger")
    def test_stop_flower_single_worker(self, mock_logger):
        local_worker = MagicMock(spec=LocalWorker, worker_id="single_worker")
        workers = Workers(local_workers=[local_worker])
        engine = AlgorithmExecutionEngine("req123", workers)

        process_ids = {"single_worker": {"server": 123, "client": 456}}

        with patch.object(engine, "stop_worker") as mock_stop_worker:
            engine.stop_flower(process_ids, "alg1")
            mock_stop_worker.assert_called_once_with(
                "single_worker", {"server": 123, "client": 456}, "alg1"
            )

    @patch("exareme2.controller.logger.get_request_logger")
    def test_stop_flower_multi_workers(self, mock_logger):
        local_worker1 = MagicMock(spec=LocalWorker, worker_id="local_worker1")
        local_worker2 = MagicMock(spec=LocalWorker, worker_id="local_worker2")
        global_worker = MagicMock(spec=GlobalWorker, worker_id="global_worker")
        workers = Workers(
            local_workers=[local_worker1, local_worker2], global_worker=global_worker
        )
        engine = AlgorithmExecutionEngine("req123", workers)

        process_ids = {
            "local_worker1": {"client": 234},
            "local_worker2": {"client": 345},
            "global_worker": {"server": 123},
        }

        with patch.object(engine, "stop_worker") as mock_stop_worker:
            engine.stop_flower(process_ids, "alg1")

            # Assert that stop_worker is called for each worker with the correct IDs
            expected_calls = [
                call("global_worker", {"server": 123}, "alg1"),
                call("local_worker1", {"client": 234}, "alg1"),
                call("local_worker2", {"client": 345}, "alg1"),
            ]
            mock_stop_worker.assert_has_calls(expected_calls, any_order=True)

    @patch("exareme2.controller.logger.get_request_logger")
    def test_stop_worker(self, mock_logger):
        local_worker = MagicMock(spec=LocalWorker, worker_id="local_worker")
        workers = Workers(local_workers=[local_worker])
        engine = AlgorithmExecutionEngine("req123", workers)

        process_ids = {"server": 1234, "client": 5678}

        engine.stop_worker("local_worker", process_ids, "alg1")
        local_worker.stop_flower_server.assert_called_once_with(1234, "alg1")
        local_worker.stop_flower_client.assert_called_once_with(5678, "alg1")

    @patch("exareme2.controller.logger.get_request_logger")
    def test_get_worker_by_id(self, mock_logger):
        local_worker1 = MagicMock(spec=LocalWorker, worker_id="local_worker1")
        local_worker2 = MagicMock(spec=LocalWorker, worker_id="local_worker2")
        global_worker = MagicMock(spec=GlobalWorker, worker_id="global_worker")
        workers = Workers(
            local_workers=[local_worker1, local_worker2], global_worker=global_worker
        )
        engine = AlgorithmExecutionEngine("req123", workers)

        # Test retrieving local worker by ID
        result = engine.get_worker_by_id("local_worker2")
        self.assertEqual(result, local_worker2)

        # Test retrieving global worker by ID
        result = engine.get_worker_by_id("global_worker")
        self.assertEqual(result, global_worker)

        # Test retrieving non-existent worker
        result = engine.get_worker_by_id("non_existent_worker")
        self.assertIsNone(result)

    @patch("exareme2.controller.logger.get_request_logger")
    def test_request_and_context_ids(self, mock_logger):
        local_worker = MagicMock(
            spec=LocalWorker,
            worker_id="local_worker",
            request_id="req123",
            context_id="ctx001",
        )
        workers = Workers(local_workers=[local_worker])
        engine = AlgorithmExecutionEngine("req123", workers)

        self.assertEqual(engine.request_id, "req123")
        self.assertEqual(engine.context_id, "ctx001")
