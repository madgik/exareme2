import enum
import traceback

from quart import Blueprint

from exareme2.controller.logger import get_background_service_logger
from exareme2.controller.services.api.algorithm_request_validator import BadRequest
from exareme2.controller.services.exareme2.controller import WorkerTaskTimeoutException
from exareme2.controller.services.exareme2.controller import WorkerUnresponsiveException
from exareme2.data_filters import FilterError
from exareme2.smpc_cluster_communication import SMPCUsageError
from exareme2.worker_communication import BadUserInput
from exareme2.worker_communication import DataModelUnavailable
from exareme2.worker_communication import DatasetUnavailable
from exareme2.worker_communication import InsufficientDataError

error_handlers = Blueprint("error_handlers", __name__)

INSUFFICIENT_DATA_ERROR_MESSAGE = (
    "The algorithm could not run with the input "
    "provided because there are insufficient data."
)


class HTTPStatusCode(enum.IntEnum):
    OK = 200
    BAD_REQUEST = 400
    BAD_USER_INPUT = 460
    INSUFFICIENT_DATA_ERROR = 461
    SMPC_USAGE_ERROR = 462
    WORKER_UNRESPONSIVE_ALGORITHM_EXECUTION_ERROR = 512
    WORKER_TASK_TIMEOUT_ALGORITHM_EXECUTION_ERROR = 513
    UNEXPECTED_ERROR = 500


@error_handlers.app_errorhandler(BadRequest)
def handle_bad_request(error: BadRequest):
    get_background_service_logger().info(
        f"Request Error. Type: '{type(error).__name__}' Message: '{error}'"
    )
    return error.message, HTTPStatusCode.BAD_REQUEST


@error_handlers.app_errorhandler(FilterError)
def handle_bad_request(error: FilterError):
    get_background_service_logger().info(
        f"Request Error. Type: '{type(error).__name__}' Message: '{error}'"
    )
    return error.message, HTTPStatusCode.BAD_REQUEST


@error_handlers.app_errorhandler(BadUserInput)
def handle_bad_user_input(error: BadUserInput):
    get_background_service_logger().info(
        f"Request Error. Type: '{type(error).__name__}' Message: '{error}'"
    )
    return error.message, HTTPStatusCode.BAD_USER_INPUT


@error_handlers.app_errorhandler(DataModelUnavailable)
def handle_bad_user_input(error: DataModelUnavailable):
    get_background_service_logger().info(
        f"Request Error. Type: '{type(error).__name__}' Message: '{error}'"
    )
    return error.message, HTTPStatusCode.BAD_USER_INPUT


@error_handlers.app_errorhandler(DatasetUnavailable)
def handle_bad_user_input(error: DatasetUnavailable):
    get_background_service_logger().info(
        f"Request Error. Type: '{type(error).__name__}' Message: '{error}'"
    )
    return error.message, HTTPStatusCode.BAD_USER_INPUT


@error_handlers.app_errorhandler(InsufficientDataError)
def handle_privacy_error(error: InsufficientDataError):
    get_background_service_logger().info(
        f"Request Error. Type: '{type(error).__name__}' Message: '{error}'"
    )
    return INSUFFICIENT_DATA_ERROR_MESSAGE, HTTPStatusCode.INSUFFICIENT_DATA_ERROR


@error_handlers.app_errorhandler(SMPCUsageError)
def handle_smpc_error(error: SMPCUsageError):
    get_background_service_logger().info(
        f"Request Error. Type: '{type(error).__name__}' Message: '{error}'"
    )
    return error.message, HTTPStatusCode.SMPC_USAGE_ERROR


@error_handlers.app_errorhandler(WorkerUnresponsiveException)
def handle_worker_unresponsive_algorithm_excecution_exception(
    error: WorkerUnresponsiveException,
):
    get_background_service_logger().error(
        f"Internal Server Error. Type: '{type(error).__name__}' Message: '{error}'"
    )
    return (
        error.message,
        HTTPStatusCode.WORKER_UNRESPONSIVE_ALGORITHM_EXECUTION_ERROR,
    )


@error_handlers.app_errorhandler(WorkerTaskTimeoutException)
def handle_worker_task_timeout_algorithm_execution_exception(
    error: WorkerTaskTimeoutException,
):
    get_background_service_logger().error(
        f"Internal Server Error. Type: '{type(error).__name__}' Message: '{error}'"
    )
    return (
        error.message,
        HTTPStatusCode.WORKER_TASK_TIMEOUT_ALGORITHM_EXECUTION_ERROR,
    )


#
# @error_handlers.app_errorhandler(Exception)
# def handle_unexpected_exception(error: Exception):
#     get_background_service_logger().error(
#         f"Internal Server Error. Type: '{type(error).__name__}' Message: '{error}'"
#     )
#     traceback.print_tb(error.__traceback__)
#     return "", HTTPStatusCode.UNEXPECTED_ERROR
