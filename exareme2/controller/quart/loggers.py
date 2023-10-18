from exareme2.controller import BACKGROUND_LOGGER_NAME
from exareme2.controller import config as ctrl_config

loggers = {
    "version": 1,
    "formatters": {
        "controller_background_service_frm": {
            "format": "%(asctime)s - %(levelname)s - CONTROLLER - BACKGROUND - %(module)s - %(funcName)s(%(lineno)d) - %(message)s",
        },
        "framework": {
            "format": "%(asctime)s - %(levelname)s - CONTROLLER - WEBAPI - %(message)s"
        },
    },
    "handlers": {
        "controller_background_service_hdl": {
            "level": ctrl_config.log_level,
            "formatter": "controller_background_service_frm",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "framework": {
            "level": ctrl_config.framework_log_level,
            "formatter": "framework",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        BACKGROUND_LOGGER_NAME: {
            "level": ctrl_config.log_level,
            "handlers": ["controller_background_service_hdl"],
        },
        "hypercorn.access": {
            "level": ctrl_config.framework_log_level,
            "handlers": ["framework"],
        },
        "hypercorn.error": {
            "level": ctrl_config.framework_log_level,
            "handlers": ["framework"],
        },
        "quart.app": {
            "level": ctrl_config.framework_log_level,
            "handlers": ["framework"],
        },
        "quart.serving": {
            "level": ctrl_config.framework_log_level,
            "handlers": ["framework"],
        },
    },
}
