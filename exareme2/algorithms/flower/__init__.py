import logging
import os

from flwr.common.logger import FLOWER_LOGGER

for handler in FLOWER_LOGGER.handlers:
    FLOWER_LOGGER.removeHandler(handler)

FLOWER_LOGGER.setLevel(logging.DEBUG)

request_id = os.getenv("REQUEST_ID", "NO-REQUEST_ID")
worker_role = os.getenv("WORKER_ROLE", "NO-ROLE")
worker_identifier = os.getenv("WORKER_IDENTIFIER", "NO-IDENTIFIER")

flower_formatter = logging.Formatter(
    f"%(asctime)s - %(levelname)s - FLOWER - {worker_role} - {worker_identifier} - %(module)s - %(funcName)s(%(lineno)d) - {request_id} - %(message)s"
)

# Configure console logger
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(flower_formatter)
FLOWER_LOGGER.addHandler(console_handler)
