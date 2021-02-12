import configparser
import importlib.resources as pkg_resources

from mipengine.node import config


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Config(metaclass=Singleton):
    def __init__(self):
        self.config = configparser.ConfigParser()
        config_str = pkg_resources.read_text(config, 'config.ini')
        self.config.read_string(config_str)
