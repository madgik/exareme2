import configparser
import sys

parser = configparser.ConfigParser()
parser.read("./mipengine/node/config/config.ini")
parser.set("monet_db", "port", sys.argv[1])
parser.set("rabbitmq", "port", sys.argv[2])
with open("./mipengine/node/config/config.ini", "w") as f:
    parser.write(f)
