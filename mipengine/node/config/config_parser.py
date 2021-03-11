from configparser import ConfigParser
import importlib.resources as pkg_resources

from mipengine.node import config as config_package

config = ConfigParser()
config_str = pkg_resources.read_text(config_package, 'config.ini')
config.read_string(config_str)