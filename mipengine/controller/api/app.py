import logging


from quart import Quart
from quart.logging import serving_handler

from mipengine.controller.api.algorithms_endpoint import algorithms
from mipengine.controller.api.error_handlers import error_handlers

app = Quart(__name__)
app.register_blueprint(algorithms)
app.register_blueprint(error_handlers)

serving_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - CONTROLLER - WEBAPI - %(message)s")
)
