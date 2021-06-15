from quart import Quart

from mipengine.controller.api.error_handlers import error_handlers
from mipengine.controller.api.algorithms_endpoint import algorithms

app = Quart(__name__)
app.register_blueprint(algorithms)
app.register_blueprint(error_handlers)
