from quart import Quart

from mipengine.controller.api.endpoint import algorithms
from mipengine.controller.api.error_handlers import error_handlers

app = Quart(__name__)
app.register_blueprint(algorithms)
app.register_blueprint(error_handlers)
