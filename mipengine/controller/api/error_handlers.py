import logging

from quart import Blueprint

from mipengine.controller.api.exceptions import BadRequest
from mipengine.controller.api.exceptions import BadUserInput
from mipengine.controller.api.exceptions import UnexpectedError
from mipengine.node_tasks_DTOs import PrivacyError

error_handlers = Blueprint("error_handlers", __name__)

PRIVACY_ERROR_MESSAGE = "The algorithm could not run with the input provided because there are insufficient data."


@error_handlers.app_errorhandler(BadRequest)
def handle_bad_request(error: BadRequest):
    return error.message, 400


@error_handlers.app_errorhandler(BadUserInput)
def handle_bad_user_input(error: BadUserInput):
    return error.message, 460


@error_handlers.app_errorhandler(PrivacyError)
def handle_privacy_error(error: PrivacyError):
    logging.error(f"Privacy Error: \n " + error.message)
    return PRIVACY_ERROR_MESSAGE, 461


@error_handlers.app_errorhandler(UnexpectedError)
def handle_unexpected_exception(error: UnexpectedError):
    return "", 500
