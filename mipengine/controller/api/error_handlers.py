import enum
import traceback

from quart import Blueprint

from mipengine.controller.api.exceptions import BadRequest
from mipengine.controller.api.exceptions import BadUserInput
from mipengine.node_tasks_DTOs import InsufficientDataError
from mipengine.controller.algorithm_executor import AlgorithmExecutionException
from mipengine.controller import controller_logger as ctrl_logger

error_handlers = Blueprint("error_handlers", __name__)

INSUFFICIENT_DATA_ERROR_MESSAGE = "The algorithm could not run with the input provided because there are insufficient data."


class HTTPStatusCode(enum.IntEnum):
    BAD_REQUEST = 400
    BAD_USER_INPUT = 460
    INSUFFICIENT_DATA_ERROR = 461
    UNEXPECTED_ERROR = 500


@error_handlers.app_errorhandler(BadRequest)
def handle_bad_request(error: BadRequest):
    return error.message, HTTPStatusCode.BAD_REQUEST


@error_handlers.app_errorhandler(BadUserInput)
def handle_bad_user_input(error: BadUserInput):
    return error.message, HTTPStatusCode.BAD_USER_INPUT


@error_handlers.app_errorhandler(InsufficientDataError)
def handle_privacy_error(error: InsufficientDataError):
    ctrl_logger.getLogger(__name__).info(
        f"Insufficient Data Error: \n " + error.message
    )
    return INSUFFICIENT_DATA_ERROR_MESSAGE, HTTPStatusCode.INSUFFICIENT_DATA_ERROR


@error_handlers.app_errorhandler(Exception)
def handle_unexpected_exception(error: Exception):
    ctrl_logger.getLogger(__name__).error(
        f"Internal Server Error."
        f"\nErrorType: {type(error)}"
        f"\nError: {error}"
        f"\nTraceback: {traceback.print_tb(error.__traceback__)}"
    )

    return "", HTTPStatusCode.UNEXPECTED_ERROR
