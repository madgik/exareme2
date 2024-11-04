import logging
import os

from flwr.common.logger import FLOWER_LOGGER

node_identifier = os.getenv("WORKER_IDENTIFIER", "NO-IDENTIFIER")
federation = os.getenv("FEDERATION", "NO-FEDERATION")
worker_role = os.getenv("WORKER_ROLE", "NO-ROLE")
framework_log_level = os.getenv("FRAMEWORK_LOG_LEVEL", "INFO")
request_id = os.getenv("REQUEST_ID", "NO-REQUEST_ID")

flower_formatter = logging.Formatter(
    f"%(asctime)s - %(levelname)s - %(module)s.%(funcName)s(%(lineno)d) - [{federation}] - [exareme2-flower-{worker_role.lower()}] - [{node_identifier}] - [{request_id}] - %(message)s"
)

# Configure console logger
console_handler = logging.StreamHandler()
console_handler.setLevel(framework_log_level)
console_handler.setFormatter(flower_formatter)

for handler in FLOWER_LOGGER.handlers:
    FLOWER_LOGGER.removeHandler(handler)
FLOWER_LOGGER.setLevel(framework_log_level)
FLOWER_LOGGER.addHandler(console_handler)
