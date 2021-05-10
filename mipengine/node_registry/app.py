from quart import Quart
from quart import request
from requests import codes

from node_registry import NodeRegistry
from mipengine.common.node_registry_DTOs import NodeRecord


app = Quart(__name__)
# app.run(debug=True)
node_catalog = NodeRegistry()


@app.route("/register", methods=["POST"])
async def register_node():
    request_body = await request.data
    node_record = NodeRecord.parse_raw(request_body)
    try:
        await node_catalog.register_node(node_record)
        return str(codes.ok)  # "200"

    except Exception as exc:
        return exc.message, str(codes.conflict)  # "409"


@app.route("/nodes")
async def get_all_nodes() -> str:
    all_nodes = await node_catalog.get_all_nodes()
    return all_nodes.json()
