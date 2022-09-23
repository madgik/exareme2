from unittest.mock import patch

import pytest

from mipengine import AttrDict
from mipengine.node.monetdb_interface.monet_db_facade import db_execute_and_fetchall
from mipengine.node.node_logger import init_logger
from tests.standalone_tests.conftest import COMMON_IP
from tests.standalone_tests.conftest import MONETDB_LOCALNODETMP_NAME
from tests.standalone_tests.conftest import MONETDB_LOCALNODETMP_PORT
from tests.standalone_tests.conftest import _restart_monetdb_container


@pytest.fixture(scope="module", autouse=True)
def patch_node_logger():
    current_task = AttrDict({"request": {"id": "1234"}})

    with patch(
        "mipengine.node.node_logger.node_config",
        AttrDict(
            {
                "log_level": "DEBUG",
                "role": "localnode",
                "identifier": "localnodetmp",
            },
        ),
    ), patch(
        "mipengine.node.node_logger.task_loggers",
        {"1234": init_logger("1234")},
    ), patch(
        "mipengine.node.node_logger.current_task",
        current_task,
    ):
        yield


@pytest.fixture(autouse=True, scope="session")
def patch_node_config():
    with patch(
        "mipengine.node.monetdb_interface.monet_db_facade.node_config",
        AttrDict(
            {
                "monetdb": {
                    "ip": COMMON_IP,
                    "port": MONETDB_LOCALNODETMP_PORT,
                    "database": "db",
                    "username": "monetdb",
                    "password": "monetdb",
                },
                "celery": {
                    "tasks_timeout": 60,
                    "run_udf_task_timeout": 120,
                    "worker_concurrency": 1,
                },
            }
        ),
    ):
        yield


@pytest.mark.slow
def test_execute_and_fetchall_success(
    monetdb_localnodetmp,
):
    result = db_execute_and_fetchall(query="SELECT 1;")
    assert result[0][0] == 1


@pytest.mark.slow
def test_broken_pipe_error_properly_handled(
    capfd,
    monetdb_localnodetmp,
):
    result = db_execute_and_fetchall(query="SELECT 1;")
    assert result[0][0] == 1
    _restart_monetdb_container(MONETDB_LOCALNODETMP_NAME)
    result = db_execute_and_fetchall(query="SELECT 1;")
    out, err = capfd.readouterr()
    assert "Trying to recover from BrokenPipeError" in err
    assert result[0][0] == 1


@pytest.mark.slow
def test_broken_pipe_error_properly_raised(
    monetdb_localnodetmp,
):
    result = db_execute_and_fetchall(query="SELECT 1;")
    assert result[0][0] == 1
    _restart_monetdb_container(MONETDB_LOCALNODETMP_NAME)

    with pytest.raises(BrokenPipeError), patch(
        "mipengine.node.monetdb_interface.monet_db_facade.BROKEN_PIPE_MAX_ATTEMPTS", 1
    ):
        db_execute_and_fetchall(query="SELECT 1;")


def test_generic_exception_handled():
    with pytest.raises(OSError):
        db_execute_and_fetchall(query="SELECT 1;")
