import os
import signal
import unittest
from unittest import mock
from unittest.mock import MagicMock
from unittest.mock import patch

import psutil

from exareme2.algorithms.flower.process_manager import ALGORITHMS_ROOT
from exareme2.algorithms.flower.process_manager import FlowerProcess
from exareme2.algorithms.flower.process_manager import handle_zombie
from exareme2.algorithms.flower.process_manager import terminate_process


class TestFlowerProcess(unittest.TestCase):
    @patch("exareme2.algorithms.flower.process_manager.subprocess.Popen")
    def test_start_process(self, mock_popen):
        """Test starting a process successfully."""
        process = FlowerProcess("script.py")
        logger = MagicMock()
        mock_popen.return_value.pid = 12345

        expected_script_path = os.path.join(ALGORITHMS_ROOT, "script.py")

        # Starting the process
        pid = process.start(logger)

        # Construct the expected command
        expected_command = (
            f"Executing command: ['poetry', 'run', 'python', '{expected_script_path}']"
        )

        # Assert the expected command is logged
        logger.info.assert_called_with(expected_command)
        self.assertEqual(pid, 12345)

    @patch("exareme2.algorithms.flower.process_manager.process_status")
    @patch("exareme2.algorithms.flower.process_manager.psutil.Process")
    def test_terminate_process(self, mock_psutil_process, mock_process_status):
        """Test terminating a process."""
        mock_process = MagicMock()
        mock_psutil_process.return_value = mock_process
        logger = MagicMock()

        # Setup the process status to simulate process still running initially then stopping
        mock_process_status.side_effect = [
            "running",  # Status before SIGTERM
            "not running",  # Status after SIGTERM
        ]

        # Call the function to test
        terminate_process(mock_process, logger)

        # Check that terminate and wait were called
        mock_process.terminate.assert_called()
        mock_process.wait.assert_called()

    @patch("exareme2.algorithms.flower.process_manager.os.waitpid")
    @patch("exareme2.algorithms.flower.process_manager.psutil.Process")
    def test_handle_zombie(self, mock_psutil_process, mock_waitpid):
        """Test handling of a zombie process."""
        # Setup the mocked process
        mock_process = MagicMock()
        mock_process.pid = 1234  # Setting the PID for the process
        mock_psutil_process.return_value = mock_process
        mock_process.status.return_value = psutil.STATUS_ZOMBIE

        # Set up the logger
        logger = MagicMock()

        # Simulate os.waitpid indicating the process has been reaped successfully
        mock_waitpid.return_value = (1234, 0)  # (pid, status)

        # Execute the function
        handle_zombie(mock_process, logger)

        # Assert that waitpid was called
        mock_waitpid.assert_called_with(1234, 0)
        logger.info.assert_called_with("Zombie process 1234 reaped successfully.")

    @patch("exareme2.algorithms.flower.process_manager.terminate_process")
    @patch("exareme2.algorithms.flower.process_manager.psutil.Process")
    def test_kill_process_class_method(
        self, mock_psutil_process, mock_terminate_process
    ):
        """Test the class method for killing a process based on algorithm name."""
        mock_process = MagicMock()
        mock_psutil_process.return_value = mock_process
        mock_process.cmdline.return_value = ["python", "script.py"]
        logger = MagicMock()
        FlowerProcess.kill_process(1234, "script", logger)
        mock_terminate_process.assert_called_once_with(mock_process, logger)

    @patch("exareme2.algorithms.flower.process_manager.terminate_process")
    @patch("exareme2.algorithms.flower.process_manager.psutil.Process")
    def test_kill_process_access_denied(
        self, mock_psutil_process, mock_terminate_process
    ):
        """Test handling access denied error during process kill based on the algorithm name."""
        # Create a mock process object with appropriate attributes
        mock_process = MagicMock()
        mock_psutil_process.return_value = mock_process
        mock_process.cmdline.return_value = ["python", "some_algorithm_script.py"]

        # Set up terminate_process to raise an AccessDenied exception
        mock_terminate_process.side_effect = psutil.AccessDenied(pid=1234)

        logger = MagicMock()

        # Assume PID 1234 and the algorithm name 'some_algorithm' is sufficient to identify the process
        FlowerProcess.kill_process(1234, "some_algorithm", logger)

        # Assert that terminate_process was called, thus confirming the algorithm name check passed
        mock_terminate_process.assert_called_once_with(mock_process, logger)

        # Check if the error was handled and logged correctly
        logger.error.assert_called_with(
            f"Access denied when attempting to terminate PID 1234."
        )
