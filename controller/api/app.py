from quart import Quart

from errors.error_handlers import error_handlers
from views.algorithms import algorithms

app = Quart(__name__)
app.register_blueprint(algorithms)
app.register_blueprint(error_handlers)
