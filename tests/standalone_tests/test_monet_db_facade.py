from unittest.mock import patch

import pytest
from pymonetdb import DatabaseError
from pymonetdb import OperationalError
from pymonetdb import ProgrammingError

from exareme2 import AttrDict
from exareme2.node.monetdb_interface.monet_db_facade import _DBExecutionDTO
from exareme2.node.monetdb_interface.monet_db_facade import _execute_and_fetchall
from exareme2.node.monetdb_interface.monet_db_facade import (
    _validate_exception_could_be_recovered,
)
from exareme2.node.node_logger import init_logger
from tests.standalone_tests.conftest import COMMON_IP
from tests.standalone_tests.conftest import MONETDB_LOCALNODETMP_NAME
from tests.standalone_tests.conftest import MONETDB_LOCALNODETMP_PORT
from tests.standalone_tests.conftest import remove_monetdb_container
from tests.standalone_tests.conftest import restart_monetdb_container


@pytest.fixture(scope="module", autouse=True)
def patch_node_logger():
    current_task = AttrDict({"request": {"id": "1234"}})

    with patch(
        "exareme2.node.node_logger.node_config",
        AttrDict(
            {
                "log_level": "DEBUG",
                "role": "localnode",
                "identifier": "localnodetmp",
            },
        ),
    ), patch(
        "exareme2.node.node_logger.task_loggers",
        {"1234": init_logger("1234")},
    ), patch(
        "exareme2.node.node_logger.current_task",
        current_task,
    ):
        yield


@pytest.fixture(autouse=True, scope="module")
def patch_node_config():
    with patch(
        "exareme2.node.monetdb_interface.monet_db_facade.node_config",
        AttrDict(
            {
                "monetdb": {
                    "ip": COMMON_IP,
                    "port": MONETDB_LOCALNODETMP_PORT,
                    "database": "db",
                    "local_username": "executor",
                    "local_password": "executor",
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
@pytest.mark.very_slow
def test_execute_and_fetchall_success(
    monetdb_localnodetmp,
):
    db_execution_dto = _DBExecutionDTO(query="select 1;", timeout=1)
    result = _execute_and_fetchall(db_execution_dto=db_execution_dto)
    assert result[0][0] == 1


@pytest.mark.slow
@pytest.mark.very_slow
def test_broken_pipe_error_properly_handled(
    capfd,
    monetdb_localnodetmp,
):
    db_execution_dto = _DBExecutionDTO(query="select 1;", timeout=1)
    result = _execute_and_fetchall(db_execution_dto=db_execution_dto)
    assert result[0][0] == 1
    restart_monetdb_container(MONETDB_LOCALNODETMP_NAME)
    db_execution_dto = _DBExecutionDTO(query="select 1;", timeout=1)
    result = _execute_and_fetchall(db_execution_dto=db_execution_dto)
    assert result[0][0] == 1


@pytest.mark.slow
@pytest.mark.very_slow
def test_generic_exception_handled(
    monetdb_localnodetmp,
):
    remove_monetdb_container(MONETDB_LOCALNODETMP_NAME)
    db_execution_dto = _DBExecutionDTO(query="select 1;", timeout=1)
    # MonetDB is inaccessible, therefore an OSError is raised.
    with pytest.raises(OSError):
        _execute_and_fetchall(db_execution_dto=db_execution_dto)


@pytest.mark.slow
@pytest.mark.very_slow
# Fix for the task https://team-1617704806227.atlassian.net/browse/MIP-731
def test_connection_error_while_waiting_for_table_to_be_present(monetdb_localnodetmp):
    db_execution_dto = _DBExecutionDTO(
        query="select * from non_existing_table;", timeout=1
    )
    with pytest.raises(OperationalError):
        _execute_and_fetchall(db_execution_dto=db_execution_dto)


def get_exception_cases():
    return [
        pytest.param(
            OSError("OSError"),
            False,
            id="Monetdb is down",
        ),
        pytest.param(
            BrokenPipeError("error"),
            True,
            id="broken pipe occurs on container restart",
        ),
        pytest.param(
            DatabaseError("table not found"),
            True,
            id="generic DatabaseError",
        ),
        pytest.param(
            DatabaseError("python exception"),
            True,
            id="python exception DatabaseError",
        ),
        pytest.param(
            DatabaseError("python exception: ValueError(print stuff)"),
            False,
            id="Algorithms debugging",
        ),
        pytest.param(
            ProgrammingError("ProgrammingError"),
            False,
            id="generic ProgrammingError",
        ),
        pytest.param(
            OperationalError(
                "42000!CREATE TABLE: insufficient privileges for user 'executor' in schema 'guest'"
            ),
            False,
            id="Insufficient privileges Error",
        ),
    ]


@pytest.mark.parametrize(
    "exception,expected",
    get_exception_cases(),
)
def test_validate_exception_is_recoverable(exception: Exception, expected):
    assert _validate_exception_could_be_recovered(exception) == expected
