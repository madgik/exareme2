import enum
import traceback

from quart import Blueprint

from mipengine.controller.api.exceptions import BadRequest
from mipengine.controller.api.exceptions import BadUserInput
from mipengine.node_tasks_DTOs import InsufficientDataError
from mipengine.controller.algorithm_executor import AlgorithmExecutionException

error_handlers = Blueprint("error_handlers", __name__)

INSUFFICIENT_DATA_ERROR_MESSAGE = "The algorithm could not run with the input provided because there are insufficient data."


class HTTPStatusCode(enum.IntEnum):
    BAD_REQUEST = 400
    BAD_USER_INPUT = 460
    INSUFFICIENT_DATA_ERROR = 461
    ALGORITHM_EXECUTION_ERROR = 462
    UNEXPECTED_ERROR = 500


@error_handlers.app_errorhandler(AlgorithmExecutionException)
def handle_algorithm_execution_exception(error: AlgorithmExecutionException):
    return error.message, HTTPStatusCode.ALGORITHM_EXECUTION_ERROR


@error_handlers.app_errorhandler(BadRequest)
def handle_bad_request(error: BadRequest):
    return error.message, HTTPStatusCode.BAD_REQUEST


@error_handlers.app_errorhandler(BadUserInput)
def handle_bad_user_input(error: BadUserInput):
    return error.message, HTTPStatusCode.BAD_USER_INPUT


@error_handlers.app_errorhandler(InsufficientDataError)
def handle_privacy_error(error: InsufficientDataError):
    print(f"Insufficient Data Error: \n " + error.message)
    return INSUFFICIENT_DATA_ERROR_MESSAGE, HTTPStatusCode.INSUFFICIENT_DATA_ERROR


@error_handlers.app_errorhandler(Exception)
def handle_unexpected_exception(error: Exception):
    print(
        f"Algorithm validation failed. \nTraceback: {traceback.print_exception(type(error), error, error.__traceback__)}"
    )
    return "", HTTPStatusCode.UNEXPECTED_ERROR
