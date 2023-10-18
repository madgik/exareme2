from quart import Quart

from exareme2.controller.quart.endpoints import algorithms
from exareme2.controller.quart.error_handlers import error_handlers

app = Quart(__name__)
app.register_blueprint(algorithms)
app.register_blueprint(error_handlers)
# app.run(debug=True)  # uncomment for breakpoints to be hit
