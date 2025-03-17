import logging
import os

from flwr.common.logger import FLOWER_LOGGER
from pythonjsonlogger import jsonlogger

node_identifier = os.getenv("WORKER_IDENTIFIER", "NO-IDENTIFIER")
federation = os.getenv("FEDERATION", "NO-FEDERATION")
worker_role = os.getenv("WORKER_ROLE", "NO-ROLE")
framework_log_level = os.getenv("FRAMEWORK_LOG_LEVEL", "INFO")
request_id = os.getenv("REQUEST_ID", "NO-REQUEST_ID")


class FlowerJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        # Populate the base fields
        super().add_fields(log_record, record, message_dict)
        # Ensure timestamp is present and named "timestamp"
        if not log_record.get("timestamp"):
            log_record["timestamp"] = self.formatTime(record, self.datefmt)
        # Override or add custom fields
        log_record["framework"] = "CELERY FRAMEWORK"
        log_record["federation"] = federation
        log_record["worker_role"] = f"exareme2-flower-{worker_role.lower()}"
        log_record["worker_identifier"] = node_identifier
        log_record["tag"] = "FRAMEWORK"


# Define a format string that includes basic fields.
# (Extra fields will be added in add_fields.)
format_str = "%(timestamp)s %(levelname)s %(message)s"
json_formatter = FlowerJsonFormatter(format_str)

# Set up the console handler with the JSON formatter.
console_handler = logging.StreamHandler()
console_handler.setLevel(framework_log_level)
console_handler.setFormatter(json_formatter)

# Remove any existing handlers from FLOWER_LOGGER and add the new handler.
for handler in FLOWER_LOGGER.handlers:
    FLOWER_LOGGER.removeHandler(handler)

FLOWER_LOGGER.setLevel(framework_log_level)
FLOWER_LOGGER.addHandler(console_handler)
