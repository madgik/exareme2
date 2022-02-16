import enum

from quart import Blueprint

from mipengine.controller.algorithm_executor import (
    NodeUnresponsiveAlgorithmExecutionException,
)
from mipengine.controller.api.exceptions import BadRequest
from mipengine.controller.api.exceptions import BadUserInput
from mipengine.filters import FilterError
from mipengine.node_tasks_DTOs import InsufficientDataError
from mipengine.smpc_cluster_comm_helpers import SMPCUsageError
from mipengine.controller import controller_logger as ctrl_logger


error_handlers = Blueprint("error_handlers", __name__)

INSUFFICIENT_DATA_ERROR_MESSAGE = (
    "The algorithm could not run with the input "
    "provided because there are insufficient data."
)


class HTTPStatusCode(enum.IntEnum):
    BAD_REQUEST = 400
    BAD_USER_INPUT = 460
    INSUFFICIENT_DATA_ERROR = 461
    SMPC_USAGE_ERROR = 462
    NODE_UNRESPONSIVE_ALGORITHM_EXECUTION_ERROR = 512
    UNEXPECTED_ERROR = 500


@error_handlers.app_errorhandler(BadRequest)
def handle_bad_request(error: BadRequest):
    return error.message, HTTPStatusCode.BAD_REQUEST


@error_handlers.app_errorhandler(FilterError)
def handle_bad_request(error: FilterError):
    return error.message, HTTPStatusCode.BAD_REQUEST


@error_handlers.app_errorhandler(BadUserInput)
def handle_bad_user_input(error: BadUserInput):
    return error.message, HTTPStatusCode.BAD_USER_INPUT


@error_handlers.app_errorhandler(InsufficientDataError)
def handle_privacy_error(error: InsufficientDataError):
    # TODO: Add proper context id. Related JIRA issue: https://team-1617704806227.atlassian.net/browse/MIP-486
    # ctrl_logger.get_request_logger("demoContextId123").info(
    #     f"Insufficient Data Error: \n " + error.message
    # )
    return INSUFFICIENT_DATA_ERROR_MESSAGE, HTTPStatusCode.INSUFFICIENT_DATA_ERROR


@error_handlers.app_errorhandler(SMPCUsageError)
def handle_privacy_error(error: SMPCUsageError):
    return error.message, HTTPStatusCode.SMPC_USAGE_ERROR


@error_handlers.app_errorhandler(NodeUnresponsiveAlgorithmExecutionException)
def handle_node_unresponsive_algorithm_excecution_exception(
    error: NodeUnresponsiveAlgorithmExecutionException,
):
    print(f"(error_handlers::handle_algorithm_excecution_exception) {error=}")
    return (
        error.message,
        HTTPStatusCode.NODE_UNRESPONSIVE_ALGORITHM_EXECUTION_ERROR,
    )


# TODO BUG https://team-1617704806227.atlassian.net/browse/MIP-476
#  Default error handler doesn't contain enough error information.
#  It's better to propagate, the error it's at least visible
# @error_handlers.app_errorhandler(Exception)
# def handle_unexpected_exception(error: Exception):
# TODO: Add proper context id. Related JIRA issue: https://team-1617704806227.atlassian.net/browse/MIP-486

#     ctrl_logger.getRequestLogger("demoContextId123").error(
#         f"Internal Server Error."
#         f"\nErrorType: {type(error)}"
#         f"\nError: {error}"
#         f"\nTraceback: {traceback.print_tb(error.__traceback__)}"
#     )
#
#     return "", HTTPStatusCode.UNEXPECTED_ERROR
