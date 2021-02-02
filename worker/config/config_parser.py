# TODO Create a static class that loads all config variables only once (Singleton)

import configparser

from controller.utils import Singleton


class Config(metaclass=Singleton):
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('./worker/config/config.ini')
