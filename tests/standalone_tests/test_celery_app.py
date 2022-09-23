from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from celery import Celery
from celery.result import AsyncResult

from mipengine.controller.celery_app import CeleryAppFactory
from mipengine.controller.celery_app import CeleryConnectionError
from mipengine.controller.celery_app import CeleryTaskTimeoutException
from mipengine.controller.celery_app import CeleryWrapper
from mipengine.node_info_DTOs import NodeInfo
from mipengine.node_tasks_DTOs import NodeTableDTO
from mipengine.node_tasks_DTOs import UDFKeyArguments
from mipengine.node_tasks_DTOs import UDFPosArguments
from mipengine.udfgen import make_unique_func_name
from tests.algorithms.orphan_udfs import five_seconds_udf
from tests.standalone_tests.conftest import RABBITMQ_GLOBALNODE_ADDR
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODETMP_ADDR
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODETMP_NAME
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODETMP_PORT
from tests.standalone_tests.conftest import _create_rabbitmq_container
from tests.standalone_tests.conftest import create_localnodetmp_node_service
from tests.standalone_tests.conftest import is_localnodetmp_node_service_ok
from tests.standalone_tests.conftest import kill_service
from tests.standalone_tests.conftest import remove_localnodetmp_rabbitmq
from tests.standalone_tests.nodes_communication_helper import get_celery_task_signature
from tests.standalone_tests.test_udfs import create_table_with_one_column_and_ten_rows

request_id = "TestingCeleryApp"
GET_NODE_INFO_TASK_TIMEOUT = 2


def execute_get_node_info_task_and_assert_response(
    cel_app_wrapper, controller_testing_logger
):
    async_res = cel_app_wrapper.queue_task(
        task_signature=get_celery_task_signature("get_node_info"),
        logger=controller_testing_logger,
        request_id=request_id,
    )
    res = cel_app_wrapper.get_result(
        async_result=async_res,
        timeout=GET_NODE_INFO_TASK_TIMEOUT,
        logger=controller_testing_logger,
    )
    assert NodeInfo.parse_raw(res)


def execute_task_and_assert_connection_error_raised(
    cel_app_wrapper, controller_testing_logger
):
    # Due to a bug on the rpc, the rabbitmq cannot recover the result, only the next scheduled task will be able to.
    async_res = cel_app_wrapper.queue_task(
        task_signature=get_celery_task_signature("get_node_info"),
        logger=controller_testing_logger,
        request_id=request_id,
    )
    with pytest.raises(CeleryConnectionError):
        cel_app_wrapper.get_result(
            async_result=async_res,
            timeout=GET_NODE_INFO_TASK_TIMEOUT,
            logger=controller_testing_logger,
        )


def queue_slow_udf(cel_app, logger):
    run_udf_task = get_celery_task_signature("run_udf")
    input_table_name, _ = create_table_with_one_column_and_ten_rows(cel_app, request_id)
    kw_args_str = UDFKeyArguments(
        args={"table": NodeTableDTO(value=input_table_name)}
    ).json()

    return cel_app.queue_task(
        task_signature=run_udf_task,
        logger=logger,
        request_id=request_id,
        command_id="1",
        context_id=request_id,
        func_name=make_unique_func_name(five_seconds_udf),
        positional_args_json=UDFPosArguments(args=[]).json(),
        keyword_args_json=kw_args_str,
        share_outputs=[True],
    )


@pytest.mark.slow
@pytest.mark.very_slow
def test_celery_app_is_the_same_after_executing_task(
    globalnode_node_service,
    reset_celery_app_factory,
    get_controller_testing_logger,
):
    cel_app_wrapper = CeleryAppFactory().get_celery_app(RABBITMQ_GLOBALNODE_ADDR)
    initial_cel_app = cel_app_wrapper._celery_app

    execute_get_node_info_task_and_assert_response(
        cel_app_wrapper, get_controller_testing_logger
    )

    assert (
        initial_cel_app
        == CeleryAppFactory().get_celery_app(RABBITMQ_GLOBALNODE_ADDR)._celery_app
    ), "Celery app is different after a queue/get of a task, even though the node never went down."


@pytest.mark.slow
@pytest.mark.very_slow
def test_celery_app_is_the_same_after_getting_slow_task_result_causing_timeout(
    globalnode_node_service,
    reset_celery_app_factory,
    get_controller_testing_logger,
):
    cel_app_wrapper = CeleryAppFactory().get_celery_app(RABBITMQ_GLOBALNODE_ADDR)
    initial_cel_app = cel_app_wrapper._celery_app

    async_res = queue_slow_udf(cel_app_wrapper, get_controller_testing_logger)

    with pytest.raises(CeleryTaskTimeoutException):
        cel_app_wrapper.get_result(
            async_result=async_res, timeout=1, logger=get_controller_testing_logger
        )

    assert (
        initial_cel_app
        == CeleryAppFactory().get_celery_app(RABBITMQ_GLOBALNODE_ADDR)._celery_app
    ), "Celery app is different after the task timed out, even though the node never went down."


@pytest.mark.slow
@pytest.mark.very_slow
def test_celery_app_is_the_same_after_get_task_result_with_exception(
    globalnode_node_service,
    reset_celery_app_factory,
    get_controller_testing_logger,
):
    cel_app_wrapper = CeleryAppFactory().get_celery_app(RABBITMQ_GLOBALNODE_ADDR)
    initial_cel_app = cel_app_wrapper._celery_app

    # Queue get node info task without providing the `request_id` thus causing an IndexError
    async_res = cel_app_wrapper.queue_task(
        task_signature=get_celery_task_signature("get_node_info"),
        logger=get_controller_testing_logger,
    )
    with pytest.raises(IndexError):
        cel_app_wrapper.get_result(
            async_result=async_res,
            timeout=10,  # A result with an error requires a bit more timeout, otherwise a TaskTimeout is thrown.
            logger=get_controller_testing_logger,
        )

    assert (
        initial_cel_app
        == CeleryAppFactory().get_celery_app(RABBITMQ_GLOBALNODE_ADDR)._celery_app
    ), "Celery app is different after the task threw an exception, even though the node never went down."


@pytest.mark.slow
@pytest.mark.very_slow
def test_celery_app_is_different_after_queue_task_when_rabbitmq_is_down(
    reset_celery_app_factory,
    get_controller_testing_logger,
):
    cel_app_wrapper = CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)
    initial_cel_app = cel_app_wrapper._celery_app

    with pytest.raises(CeleryConnectionError):
        cel_app_wrapper.queue_task(
            task_signature=get_celery_task_signature("get_node_info"),
            logger=get_controller_testing_logger,
            request_id=request_id,
        )

    assert (
        initial_cel_app
        != CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)._celery_app
    ), "The celery apps should NOT be the same, the rabbitmq is down causing a reset."

    assert isinstance(
        CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)._celery_app,
        Celery,
    ), "The new celery app is not an instance of Celery. Something unexpected occurred during the reset."


@pytest.mark.slow
@pytest.mark.very_slow
def test_celery_app_is_different_after_get_task_res_when_rabbitmq_is_down(
    reset_celery_app_factory,
    get_controller_testing_logger,
):
    cel_app_wrapper = CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)
    initial_cel_app = cel_app_wrapper._celery_app

    with pytest.raises(CeleryConnectionError):
        cel_app_wrapper.get_result(
            async_result=AsyncResult(app=cel_app_wrapper._celery_app, id="whatever"),
            timeout=10,
            logger=get_controller_testing_logger,
        )

    assert (
        initial_cel_app
        != CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)._celery_app
    ), "The celery app should reset, there was a connectivity error."

    assert isinstance(
        CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)._celery_app,
        Celery,
    ), "The new celery app is not an instance of Celery. Something unexpected occurred during the reset."


@pytest.mark.slow
@pytest.mark.very_slow
def test_celery_app_is_the_same_after_get_task_res_with_node_down(
    localnodetmp_node_service,
    reset_celery_app_factory,
    get_controller_testing_logger,
):
    cel_app_wrapper = CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)
    initial_cel_app = cel_app_wrapper._celery_app

    # We are queuing a slow udf here so that it doesn't quickly complete before we kill the service.
    async_res = queue_slow_udf(cel_app_wrapper, get_controller_testing_logger)

    kill_service(localnodetmp_node_service)

    with pytest.raises(CeleryTaskTimeoutException):
        cel_app_wrapper.get_result(
            async_result=async_res,
            timeout=GET_NODE_INFO_TASK_TIMEOUT,
            logger=get_controller_testing_logger,
        )

    assert (
        initial_cel_app
        == CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)._celery_app
    ), "The celery apps should be the same, the rabbitmq never went down."


@pytest.mark.slow
@pytest.mark.very_slow
def test_celery_app_is_the_same_after_getting_task_when_node_restarted(
    localnodetmp_node_service,
    reset_celery_app_factory,
    get_controller_testing_logger,
):
    cel_app_wrapper = CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)
    initial_cel_app = cel_app_wrapper._celery_app

    # Restart the node service
    kill_service(localnodetmp_node_service)
    restarted_localnodetmp_node_service = create_localnodetmp_node_service()

    execute_get_node_info_task_and_assert_response(
        cel_app_wrapper, get_controller_testing_logger
    )

    assert (
        initial_cel_app
        == CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)._celery_app
    ), "The celery apps should be the same, the rabbitmq never went down."

    kill_service(restarted_localnodetmp_node_service)


@pytest.mark.slow
@pytest.mark.very_slow
def test_celery_app_is_different_after_get_result_when_rabbitmq_restarted(
    localnodetmp_node_service,
    reset_celery_app_factory,
    get_controller_testing_logger,
):
    cel_app_wrapper = CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)
    initial_cel_app = cel_app_wrapper._celery_app

    # Initialize the celery app by using it to queue a task and get its result.
    # This is needed because the celery_app result consumer doesn't recover if
    # the connection is severed.
    execute_get_node_info_task_and_assert_response(
        cel_app_wrapper, get_controller_testing_logger
    )

    # Restart the rabbitmq
    remove_localnodetmp_rabbitmq()
    _create_rabbitmq_container(RABBITMQ_LOCALNODETMP_NAME, RABBITMQ_LOCALNODETMP_PORT)

    if not is_localnodetmp_node_service_ok(localnodetmp_node_service):
        localnodetmp_node_service = create_localnodetmp_node_service()

    execute_task_and_assert_connection_error_raised(
        cel_app_wrapper, get_controller_testing_logger
    )

    assert (
        initial_cel_app
        != CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)._celery_app
    ), (
        "The celery apps should NOT be the same, the rabbitmq restarted and the broker got corrupted."
        "https://github.com/celery/celery/issues/6912#issuecomment-1107260087"
    )

    assert isinstance(
        CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)._celery_app,
        Celery,
    ), "The new celery app is not an instance of Celery. Something unexpected occurred during the reset."

    execute_get_node_info_task_and_assert_response(
        cel_app_wrapper, get_controller_testing_logger
    )

    kill_service(localnodetmp_node_service)


@pytest.mark.slow
@pytest.mark.very_slow
def test_celery_app_didnt_change_too_many_times_after_parallel_get_task_result_when_rabbitmq_restarted(
    localnodetmp_node_service,
    reset_celery_app_factory,
    get_controller_testing_logger,
):
    cel_app_wrapper = CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)
    initial_cel_app = cel_app_wrapper._celery_app

    concurrent_requests = 20

    # Initialize celery app to open channels
    execute_get_node_info_task_and_assert_response(
        cel_app_wrapper, get_controller_testing_logger
    )

    async_results = [
        cel_app_wrapper.queue_task(
            task_signature=get_celery_task_signature("get_node_info"),
            logger=get_controller_testing_logger,
            request_id=request_id,
        )
        for _ in range(concurrent_requests)
    ]

    # Restart the rabbitmq
    remove_localnodetmp_rabbitmq()
    _create_rabbitmq_container(RABBITMQ_LOCALNODETMP_NAME, RABBITMQ_LOCALNODETMP_PORT)

    if not is_localnodetmp_node_service_ok(localnodetmp_node_service):
        localnodetmp_node_service = create_localnodetmp_node_service()

    with patch.object(
        CeleryWrapper,
        "_instantiate_celery_object",
        wraps=cel_app_wrapper._instantiate_celery_object,
    ) as instantiate_celery_app_mock:
        with ThreadPoolExecutor() as executor:
            future_task_results = [
                executor.submit(
                    cel_app_wrapper.get_result,
                    async_res,
                    GET_NODE_INFO_TASK_TIMEOUT,
                    get_controller_testing_logger,
                )
                for async_res in async_results
            ]
        for future in future_task_results:
            # The result could throw an exception or return an actual value depending on
            # if the result was fetched before the rabbitmq restarted.
            try:
                future.result()
                print("No exception!")
            except Exception as exc:
                print(f"Exception {type(exc)}")

    assert 1 <= instantiate_celery_app_mock.call_count < concurrent_requests / 2, (
        "A new celery app should be created, at least once. "
        "A new celery app is created upon OSError and other errors to reset the app after the rabbitmq restart."
        "The OSError and other celery resetting errors could happen more than once, depending on race conditions."
        "The celery app should not be instantiated as many times as the requests, this would mean that we are not"
        "catching the correct errors. "
        "This formula '1 <= instantiate_celery_app_mock.call_count < concurrent_requests / 2' is the best stable "
        "check that could be achieved, balancing between NOT resetting the celery app each time and checking that"
        "it was actually reset."
    )

    assert (
        initial_cel_app
        != CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)._celery_app
    ), (
        "The celery apps should NOT be the same, the rabbitmq restarted and the broker got corrupted."
        "https://github.com/celery/celery/issues/6912#issuecomment-1107260087"
    )

    assert isinstance(
        CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)._celery_app,
        Celery,
    ), "The new celery app is not an instance of Celery. Something unexpected occurred during the reset."

    execute_get_node_info_task_and_assert_response(
        cel_app_wrapper, get_controller_testing_logger
    )

    kill_service(localnodetmp_node_service)
