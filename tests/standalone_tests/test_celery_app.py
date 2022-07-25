from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
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
from tests.algorithms.orphan_udfs import very_slow_udf
from tests.standalone_tests.conftest import RABBITMQ_GLOBALNODE_ADDR
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODETMP_ADDR
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODETMP_NAME
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODETMP_PORT
from tests.standalone_tests.conftest import _create_rabbitmq_container
from tests.standalone_tests.conftest import create_localnodetmp_node_service
from tests.standalone_tests.conftest import ensure_localnodetmp_node_service_is_running
from tests.standalone_tests.conftest import kill_service
from tests.standalone_tests.conftest import remove_localnodetmp_rabbitmq
from tests.standalone_tests.nodes_communication_helper import get_celery_task_signature
from tests.standalone_tests.test_udfs import create_table_with_one_column_and_ten_rows

request_id = "TestingCeleryApp"
GET_NODE_INFO_TASK_TIMEOUT = 2


def send_get_node_info_task_and_assert_response(
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


def reset_the_celery_app_by_submitting_a_task_that_will_fail(
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
    input_table_name, input_table_name_sum = create_table_with_one_column_and_ten_rows(
        cel_app
    )
    kw_args_str = UDFKeyArguments(
        args={"table": NodeTableDTO(value=input_table_name)}
    ).json()

    return cel_app.queue_task(
        task_signature=run_udf_task,
        logger=logger,
        request_id=request_id,
        command_id="1",
        context_id=request_id,
        func_name=make_unique_func_name(very_slow_udf),
        positional_args_json=UDFPosArguments(args=[]).json(),
        keyword_args_json=kw_args_str,
    )


@pytest.mark.slow
@pytest.mark.integration
def test_celery_app_queue_and_get_task(
    globalnode_node_service,
    reset_celery_app_factory,
    controller_testing_logger,
):
    cel_app_wrapper = CeleryAppFactory().get_celery_app(RABBITMQ_GLOBALNODE_ADDR)
    initial_cel_app = cel_app_wrapper._celery_app

    send_get_node_info_task_and_assert_response(
        cel_app_wrapper, controller_testing_logger
    )

    assert (
        initial_cel_app
        == CeleryAppFactory().get_celery_app(RABBITMQ_GLOBALNODE_ADDR)._celery_app
    ), "Celery app is different after the task was queued and fetched."


@pytest.mark.slow
@pytest.mark.integration
def test_celery_app_get_task_timeout(
    globalnode_node_service,
    reset_celery_app_factory,
    controller_testing_logger,
):
    cel_app_wrapper = CeleryAppFactory().get_celery_app(RABBITMQ_GLOBALNODE_ADDR)
    initial_cel_app = cel_app_wrapper._celery_app

    async_res = queue_slow_udf(cel_app_wrapper, controller_testing_logger)

    with pytest.raises(CeleryTaskTimeoutException):
        cel_app_wrapper.get_result(
            async_result=async_res, timeout=1, logger=controller_testing_logger
        )

    assert (
        initial_cel_app
        == CeleryAppFactory().get_celery_app(RABBITMQ_GLOBALNODE_ADDR)._celery_app
    ), "Celery app is different after the task timed out, even though the node never went down."


@pytest.mark.integration
def test_celery_app_queue_task_with_rabbitmq_down(
    reset_celery_app_factory,
    controller_testing_logger,
):
    cel_app_wrapper = CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)
    initial_cel_app = cel_app_wrapper._celery_app

    with pytest.raises(CeleryConnectionError):
        cel_app_wrapper.queue_task(
            task_signature=get_celery_task_signature("get_node_info"),
            logger=controller_testing_logger,
            request_id=request_id,
        )

    assert (
        initial_cel_app
        == CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)._celery_app
    ), "The celery apps should be the same, the rabbitmq is still unreachable."


@pytest.mark.integration
def test_celery_app_get_task_res_with_rabbitmq_down(
    reset_celery_app_factory,
    controller_testing_logger,
):
    cel_app_wrapper = CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)
    initial_cel_app = cel_app_wrapper._celery_app

    with pytest.raises(CeleryConnectionError):
        cel_app_wrapper.get_result(
            async_result=AsyncResult(app=cel_app_wrapper._celery_app, id="whatever"),
            timeout=10,
            logger=controller_testing_logger,
        )

    assert (
        initial_cel_app
        != CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)._celery_app
    ), "The celery app should reset, there was a connectivity error."


@pytest.mark.slow
@pytest.mark.integration
def test_celery_app_get_task_res_with_node_down(
    localnodetmp_node_service,
    reset_celery_app_factory,
    controller_testing_logger,
):
    cel_app_wrapper = CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)
    initial_cel_app = cel_app_wrapper._celery_app

    async_res = queue_slow_udf(cel_app_wrapper, controller_testing_logger)

    kill_service(localnodetmp_node_service)

    with pytest.raises(CeleryTaskTimeoutException):
        cel_app_wrapper.get_result(
            async_result=async_res,
            timeout=GET_NODE_INFO_TASK_TIMEOUT,
            logger=controller_testing_logger,
        )

    assert (
        initial_cel_app
        == CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)._celery_app
    ), "The celery apps should be the same, the rabbitmq never went down."


@pytest.mark.slow
@pytest.mark.integration
def test_celery_app_submit_task_after_node_restart(
    localnodetmp_node_service,
    reset_celery_app_factory,
    controller_testing_logger,
):
    cel_app_wrapper = CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)
    initial_cel_app = cel_app_wrapper._celery_app

    # Restart the node service
    kill_service(localnodetmp_node_service)
    restarted_localnodetmp_node_service = create_localnodetmp_node_service()

    send_get_node_info_task_and_assert_response(
        cel_app_wrapper, controller_testing_logger
    )

    assert (
        initial_cel_app
        == CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)._celery_app
    ), "The celery apps should be the same, the rabbitmq never went down."

    kill_service(restarted_localnodetmp_node_service)


@pytest.mark.slow
@pytest.mark.integration
def test_celery_app_broker_get_result_after_rabbitmq_restart(
    localnodetmp_node_service,
    reset_celery_app_factory,
    controller_testing_logger,
):
    cel_app_wrapper = CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)
    initial_cel_app = cel_app_wrapper._celery_app

    # Initialize celery app to open channels
    send_get_node_info_task_and_assert_response(
        cel_app_wrapper, controller_testing_logger
    )

    # Restart the rabbitmq
    remove_localnodetmp_rabbitmq()
    _create_rabbitmq_container(RABBITMQ_LOCALNODETMP_NAME, RABBITMQ_LOCALNODETMP_PORT)

    localnodetmp_node_service = ensure_localnodetmp_node_service_is_running(
        localnodetmp_node_service
    )

    reset_the_celery_app_by_submitting_a_task_that_will_fail(
        cel_app_wrapper, controller_testing_logger
    )

    assert (
        initial_cel_app
        != CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)._celery_app
    ), (
        "The celery apps should NOT be the same, the rabbitmq restarted and the broker got corrupted."
        "https://github.com/celery/celery/issues/6912#issuecomment-1107260087"
    )

    send_get_node_info_task_and_assert_response(
        cel_app_wrapper, controller_testing_logger
    )

    kill_service(localnodetmp_node_service)


@pytest.mark.slow
@pytest.mark.integration
def test_celery_app_broker_get_result_with_exception_after_rabbitmq_restart(
    localnodetmp_node_service,
    reset_celery_app_factory,
    controller_testing_logger,
):
    cel_app_wrapper = CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)
    initial_cel_app = cel_app_wrapper._celery_app

    # Initialize celery app to open channels
    send_get_node_info_task_and_assert_response(
        cel_app_wrapper, controller_testing_logger
    )

    # Restart the rabbitmq
    remove_localnodetmp_rabbitmq()
    _create_rabbitmq_container(RABBITMQ_LOCALNODETMP_NAME, RABBITMQ_LOCALNODETMP_PORT)

    localnodetmp_node_service = ensure_localnodetmp_node_service_is_running(
        localnodetmp_node_service
    )

    reset_the_celery_app_by_submitting_a_task_that_will_fail(
        cel_app_wrapper, controller_testing_logger
    )

    async_res = cel_app_wrapper.queue_task(
        task_signature=get_celery_task_signature("get_node_info"),
        logger=controller_testing_logger,
    )
    with pytest.raises(IndexError):
        cel_app_wrapper.get_result(
            async_result=async_res,
            timeout=GET_NODE_INFO_TASK_TIMEOUT,
            logger=controller_testing_logger,
        )

    assert (
        initial_cel_app
        != CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)._celery_app
    ), (
        "The celery apps should NOT be the same, the rabbitmq restarted and the broker got corrupted."
        "https://github.com/celery/celery/issues/6912#issuecomment-1107260087"
    )

    kill_service(localnodetmp_node_service)


@pytest.mark.slow
@pytest.mark.integration
def test_celery_app_parallel_submit_task_after_rabbitmq_restart(
    localnodetmp_node_service,
    reset_celery_app_factory,
    controller_testing_logger,
):
    cel_app_wrapper = CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)
    initial_cel_app = cel_app_wrapper._celery_app

    concurrent_requests = 20

    # Initialize celery app to open channels
    send_get_node_info_task_and_assert_response(
        cel_app_wrapper, controller_testing_logger
    )

    async_results = [
        cel_app_wrapper.queue_task(
            task_signature=get_celery_task_signature("get_node_info"),
            logger=controller_testing_logger,
            request_id=request_id,
        )
        for _ in range(concurrent_requests)
    ]

    # Restart the rabbitmq
    remove_localnodetmp_rabbitmq()
    _create_rabbitmq_container(RABBITMQ_LOCALNODETMP_NAME, RABBITMQ_LOCALNODETMP_PORT)

    localnodetmp_node_service = ensure_localnodetmp_node_service_is_running(
        localnodetmp_node_service
    )

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
                    controller_testing_logger,
                )
                for async_res in async_results
            ]
        for future in future_task_results:
            # The result could throw an exception or return an actual value depending on
            # if the result was fetched before the rabbitmq restarted.
            try:
                future.result()
            except (CeleryConnectionError, CeleryTaskTimeoutException):
                pass

    assert 1 <= instantiate_celery_app_mock.call_count < concurrent_requests / 2, (
        "A new celery app should be created, at least once. "
        "A new celery app is created upon OSError to reset the app after the rabbitmq restart."
        "The celery app should not be instantiated as many times as the requests. Only ypon OSError."
    )

    assert (
        initial_cel_app
        != CeleryAppFactory().get_celery_app(RABBITMQ_LOCALNODETMP_ADDR)._celery_app
    ), (
        "The celery apps should NOT be the same, the rabbitmq restarted and the broker got corrupted."
        "https://github.com/celery/celery/issues/6912#issuecomment-1107260087"
    )

    send_get_node_info_task_and_assert_response(
        cel_app_wrapper, controller_testing_logger
    )

    kill_service(localnodetmp_node_service)
