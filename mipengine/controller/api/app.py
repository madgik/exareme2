from quart import Quart

from mipengine.controller.api.error_handlers import error_handlers
from mipengine.controller.api.algorithms_endpoint import algorithms
from mipengine.controller import config as ctrl_config
from logging import Formatter
from quart.logging import serving_handler

serving_handler.setFormatter(
    Formatter(
        "%(asctime)s - %(levelname)s - CONTROLLER - QUART - %(module)s - %(message)s"
    )
)
serving_handler.setLevel(ctrl_config.log_level)

app = Quart(__name__)
app.register_blueprint(algorithms)
app.register_blueprint(error_handlers)
