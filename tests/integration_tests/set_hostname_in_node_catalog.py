import json
from argparse import ArgumentParser
from pathlib import Path

from mipengine.common import resources

parser = ArgumentParser()
parser.add_argument(
    "-host", "--hostname", required=True, help="The hostname of the monetdbs."
)

args = parser.parse_args()
hostname = args.hostname

resources_folder_path = Path(resources.__file__).parent
node_catalog_abs_path = Path(resources_folder_path, "node_catalog.json")
node_catalog_stream = open(node_catalog_abs_path, "r")

node_catalog = json.load(node_catalog_stream)
for local_node in node_catalog["localNodes"]:
    local_node["monetdbIp"] = hostname
    local_node["rabbitmqIp"] = hostname

node_catalog["globalNode"]["monetdbIp"] = hostname
node_catalog["globalNode"]["rabbitmqIp"] = hostname

with open(node_catalog_abs_path, "w") as node_catalog_file:
    json.dump(node_catalog, node_catalog_file, indent=2)

print(f"Successfully changed node catalog hostnames to '{hostname}'.")
