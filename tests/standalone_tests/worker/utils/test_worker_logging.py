import re
from unittest.mock import patch

import pytest

from exareme2 import AttrDict
from exareme2.worker.utils import logger as worker_logger

task_loggers = {}


@pytest.fixture(scope="module", autouse=True)
def mock_worker_config():
    worker_config = AttrDict(
        {
            "identifier": "localworker1",
            "federation": "standalone_tests",
            "log_level": "INFO",
            "role": "LOCALWORKER",
        }
    )

    with patch(
        "exareme2.worker.utils.logger.worker_config",
        worker_config,
    ):
        yield

    return worker_config


@pytest.fixture(scope="module", autouse=True)
def mock_current_task():
    current_task = AttrDict({"request": {"id": "1234"}})

    with patch(
        "exareme2.worker.utils.logger.current_task",
        current_task,
    ):
        yield

    return current_task


@worker_logger.initialise_logger
def pass_rqst_id(request_id):
    logger = worker_logger.get_logger()
    logger.info("Yolo!")
    return logger


def test_get_ctx_id_from_args(capsys):
    test_ctx_id = pass_rqst_id("1234abcd")
    captured = capsys.readouterr()

    # regex to check timestamp
    my_regex = re.compile(r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}\s-[^"]*')
    assert my_regex.match(captured.err) is not None
    assert captured.err.find(" INFO ") > -1
    assert captured.err.find("[exareme2-localworker]") > -1
    assert captured.err.find("[1234abcd]") > -1
    assert test_ctx_id.name == "1234abcd"
    assert test_ctx_id.level == 20


@worker_logger.initialise_logger
def initialize_and_return_logger(request_id):
    logger = worker_logger.get_logger()
    return logger


def test_get_ctx_id_from_args1(capsys):
    logger1 = initialize_and_return_logger("logger1")
    initialize_and_return_logger("logger2")

    logger1.info("Log...")

    captured = capsys.readouterr()

    # regex to check timestamp
    my_regex = re.compile(r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}\s-[^"]*')
    assert my_regex.match(captured.err) is not None
    assert captured.err.find(" INFO ") > -1
    assert captured.err.find("[exareme2-localworker]") > -1
    assert captured.err.find("[logger1]") > -1
    assert logger1.name == "logger1"
    assert logger1.level == 20
