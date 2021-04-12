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
    local_node["monetdbHostname"] = hostname

    tmp = local_node["rabbitmqURL"]
    tmp = tmp.split(":")
    tmp[0] = hostname
    local_node["rabbitmqURL"] = f"{tmp[0]}:{tmp[1]}"

node_catalog["globalNode"]["monetdbHostname"] = hostname
tmp = node_catalog["globalNode"]["rabbitmqURL"]
tmp = tmp.split(":")
tmp[0] = hostname
node_catalog["globalNode"]["rabbitmqURL"] = f"{tmp[0]}:{tmp[1]}"

with open(node_catalog_abs_path, "w") as node_catalog_file:
    json.dump(node_catalog, node_catalog_file, indent=2)

print(f"Successfully changed node catalog hostnames to '{hostname}'.")
