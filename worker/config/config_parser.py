import configparser
import importlib.resources as pkg_resources

from controller.utils import Singleton
from worker import config


class Config(metaclass=Singleton):
    def __init__(self):
        self.config = configparser.ConfigParser()
        config_str = pkg_resources.read_text(config, 'config.ini')

        self.config.read_string(config_str)
