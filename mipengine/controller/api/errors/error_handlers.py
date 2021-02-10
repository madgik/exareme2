from quart import Blueprint

from mipengine.controller.api.errors.exceptions import BadRequest, BadUserInput

error_handlers = Blueprint('error_handlers', __name__)


@error_handlers.app_errorhandler(BadRequest)
def handle_bad_request(error: BadRequest):
    return error.message, error.status_code


@error_handlers.app_errorhandler(BadUserInput)
def handle_bad_user_input(error: BadUserInput):
    return error.message, error.status_code
