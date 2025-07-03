from quart import Quart

from exareme2.controller.quart.endpoints import algorithms
from exareme2.controller.quart.error_handlers import error_handlers

#
# QUART_HOST = "0.0.0.0"
# QUART_PORT = 5000
# QUART_DEBUG = True  # Change to enable debug
#

app = Quart(__name__)
app.register_blueprint(algorithms)
app.register_blueprint(error_handlers)

# We need to pass host and port as well, otherwise the default values are used
# app.run(host=QUART_HOST, port=QUART_PORT, debug=QUART_DEBUG)
