from quart import Blueprint

from mipengine.controller.api.exceptions import BadRequest
from mipengine.controller.api.exceptions import BadUserInput
from mipengine.controller.api.exceptions import UnexpectedException

error_handlers = Blueprint("error_handlers", __name__)


@error_handlers.app_errorhandler(BadRequest)
def handle_bad_request(error: BadRequest):
    return error.message, error.status_code


@error_handlers.app_errorhandler(BadUserInput)
def handle_bad_user_input(error: BadUserInput):
    return error.message, error.status_code


@error_handlers.app_errorhandler(UnexpectedException)
def handle_unexpected_exception(error: UnexpectedException):
    return "", error.status_code
