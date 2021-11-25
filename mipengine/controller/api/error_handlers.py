import enum
import traceback

from quart import Blueprint

from mipengine.controller.api.exceptions import BadRequest
from mipengine.controller.api.exceptions import BadUserInput
from mipengine.node_tasks_DTOs import InsufficientDataError
from mipengine.controller.algorithm_executor import AlgorithmExecutionException

error_handlers = Blueprint("error_handlers", __name__)

INSUFFICIENT_DATA_ERROR_MESSAGE = (
    "The algorithm could not run with the input "
    "provided because there are insufficient data."
)

ALGORITHM_EXUCUTION_ERROR = "An error occured during the execution of the algorithm"


class HTTPStatusCode(enum.IntEnum):
    BAD_REQUEST = 400
    BAD_USER_INPUT = 460
    INSUFFICIENT_DATA_ERROR = 461
    ALGORITHM_EXECUTION_ERROR = 462
    UNEXPECTED_ERROR = 500


@error_handlers.app_errorhandler(BadRequest)
def handle_bad_request(error: BadRequest):
    return error.message, HTTPStatusCode.BAD_REQUEST


@error_handlers.app_errorhandler(BadUserInput)
def handle_bad_user_input(error: BadUserInput):
    return error.message, HTTPStatusCode.BAD_USER_INPUT


@error_handlers.app_errorhandler(InsufficientDataError)
def handle_privacy_error(error: InsufficientDataError):
    print(
        f"(error_handlers::handle_privacy_error) Insufficient Data Error: \n "
        + error.message
    )
    return INSUFFICIENT_DATA_ERROR_MESSAGE, HTTPStatusCode.INSUFFICIENT_DATA_ERROR


@error_handlers.app_errorhandler(AlgorithmExecutionException)
def handle_algorithm_excecution_exception(error: AlgorithmExecutionException):
    print(f"(error_handlers::handle_algorithm_excecution_exception) {error=}")
    return ALGORITHM_EXUCUTION_ERROR, HTTPStatusCode.ALGORITHM_EXECUTION_ERROR


@error_handlers.app_errorhandler(Exception)
def handle_unexpected_exception(error: Exception):
    import traceback

    traceback_str = "".join(traceback.format_tb(error.__traceback__))
    print(
        f"(error_handlers::handle_unexpected_exception) Unexpected Exception raised->\n{traceback_str} {error}"
    )
    return "", HTTPStatusCode.UNEXPECTED_ERROR
