import sys

from mipengine.config.config_parser import Config

parser = Config().config
parser.set("monet_db", "port", sys.argv[1])
parser.set("rabbitmq", "port", sys.argv[2])
with open("config.ini", "w") as f:
    parser.write(f)
