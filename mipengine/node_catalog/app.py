from quart import Quart
from quart import request

from node_catalog import NodeRegistry
from mipengine.common.node_catalog_DTOs import Pathology, NodeRecord

import asyncio

import concurrent.futures


app = Quart(__name__)
# app.run(debug=True)
node_catalog = NodeRegistry()


@app.route("/register", methods=["POST"])
async def register_node():
    print("(register_node)")
    request_body = await request.data
    node_record = NodeRecord.parse_raw(request_body)
    try:
        await node_catalog.register_node(node_record)
        print(f"(register_node) registered: {node_record}")
        return "Sucess", "200"

    except Exception as exc:
        print(f"(register_node) FAILED: {exc.message}")
        return exc.message, "409"


@app.route("/nodes")
async def get_all_nodes() -> str:
    print("(get_all_nodes)")
    all_nodes = await node_catalog.get_all_nodes()

    print(f"all_nodes--> {all_nodes}")

    all_nodes = [node_record.json() for node_record in all_nodes]

    import json

    str_json = json.dumps(all_nodes)
    return str_json, "200"
