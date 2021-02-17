import configparser
import sys

parser = configparser.ConfigParser()
parser.read("./mipengine/node/config/config.ini")
parser.set("node", "identifier", sys.argv[1])
with open("./mipengine/node/config/config.ini", "w") as f:
    parser.write(f)
