import pytest

from exareme2.algorithms.exareme2.udfgen import make_unique_func_name
from exareme2.controller.celery.app import CeleryAppFactory
from exareme2.controller.celery.app import CeleryTaskTimeoutException
from exareme2.controller.services.worker_info_tasks_handler import WorkerInfoTasksHandler
from exareme2.worker import config as worker_config
from exareme2.worker_communication import WorkerInfo
from exareme2.worker_communication import WorkerTableDTO
from exareme2.worker_communication import WorkerUDFKeyArguments
from exareme2.worker_communication import WorkerUDFPosArguments
from tests.algorithms.orphan_udfs import one_second_udf
from tests.standalone_tests.conftest import RABBITMQ_GLOBALWORKER_ADDR
from tests.standalone_tests.test_udfs import create_table_with_one_column_and_ten_rows
from tests.standalone_tests.workers_communication_helper import (
    get_celery_task_signature,
)

request_id = "TestingSystemCallPriority"
command_id = 0


def queue_one_second_udf(
    cel_app,
    input_table_name,
    logger,
):
    run_udf_task = get_celery_task_signature("run_udf")

    kw_args_str = WorkerUDFKeyArguments(
        args={"table": WorkerTableDTO(value=input_table_name)}
    ).json()

    global command_id
    command_id += 1
    return cel_app.queue_task(
        task_signature=run_udf_task,
        logger=logger,
        request_id=request_id,
        command_id=str(command_id),
        context_id=request_id,
        func_name=make_unique_func_name(one_second_udf),
        positional_args_json=WorkerUDFPosArguments(args=[]).json(),
        keyword_args_json=kw_args_str,
    )


@pytest.mark.slow
@pytest.mark.very_slow
def test_worker_info_tasks_have_higher_priority_over_other_tasks(
    globalworker_worker_service,
    globalworker_db_cursor,
    reset_celery_app_factory,
    get_controller_testing_logger,
):
    cel_app_wrapper = CeleryAppFactory().get_celery_app(RABBITMQ_GLOBALWORKER_ADDR)

    input_table_name, _ = create_table_with_one_column_and_ten_rows(
        cel_app_wrapper, globalworker_db_cursor, request_id
    )

    # Queue an X amount of udfs to fill the rabbitmq.
    # The queued udfs should be greater than the NODE workers that consume them.
    number_of_udfs_to_schedule = worker_config.celery.worker_concurrency + 10
    udf_async_results = [
        queue_one_second_udf(
            cel_app_wrapper, input_table_name, get_controller_testing_logger
        )
        for _ in range(number_of_udfs_to_schedule)
    ]

    # The worker info task should wait for one udf to complete (~1-2sec), in order to start being executed,
    # but shouldn't wait for more than one, since it has priority of the other celery.
    # The timeout should be the time taken for one udf to complete, plus some additional time for
    # the actual get_worker_info task to complete.
    worker_info_task_timeout = 3
    worker_info_task_handler = WorkerInfoTasksHandler(
        RABBITMQ_GLOBALWORKER_ADDR, worker_info_task_timeout, request_id
    )
    try:
        result = worker_info_task_handler.get_worker_info_task()
    except CeleryTaskTimeoutException as exc:
        pytest.fail(
            f"The worker info task should not wait for the other tasks but a timeout occurred."
            f"Exception: {exc}"
        )
    assert isinstance(result, WorkerInfo)

    for async_res in udf_async_results:
        udf_tasks_timeout = number_of_udfs_to_schedule * 2
        try:
            cel_app_wrapper.get_result(
                async_result=async_res,
                timeout=udf_tasks_timeout,
                logger=get_controller_testing_logger,
            )
        except Exception as exc:
            pytest.fail(
                f"An exception shouldn't occur. Exception Type: {type(exc)}, \n Exception: {exc}"
            )
