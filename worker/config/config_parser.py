import configparser

from controller.utils import Singleton


class Config(metaclass=Singleton):
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('./worker/config/config.ini')
