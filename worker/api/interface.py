from typing import Union, Tuple

from flask import Flask, jsonify, abort, request

from DTOs import TableIdentifier, DistributedTableIdentifier, TableSchemaTypes, UDF, TableView

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/tables/<table_name>', methods=['GET'])
def get_table(table_name: str) -> Tuple[Union[TableIdentifier, DistributedTableIdentifier], int]:
    # TODO if table_name doesn't exist or this is a local node
    if table_name == 'kostas':
        abort(400)

    # TODO Fetch table data if this is a global node
    table1 = TableIdentifier(
        table_name,
        [TableSchemaTypes.FLOAT, TableSchemaTypes.INT],
        "worker_1")

    table2 = TableIdentifier(
        table_name,
        [TableSchemaTypes.FLOAT, TableSchemaTypes.INT],
        "worker_2")

    table = DistributedTableIdentifier([table1, table2])
    return jsonify(table.serialize()), 200


@app.route('/tables/runUDF', methods=['POST'])
def run_udf():
    udf: UDF = UDF.from_json(request.data)

    # TODO Run UDF
    table_identifier: TableIdentifier = None

    return table_identifier, 201


@app.route('/tables/createView', methods=['POST'])
def create_view():
    view: TableView = TableView.from_json(request.data)

    # TODO Create view
    table_identifier: TableIdentifier = None

    return table_identifier, 201


@app.route('/tables/<table_name>', methods=['DELETE'])
def delete_table(table_name: str) -> Tuple[str, int]:
    # TODO if table_name doesn't exist or this is a local node
    if table_name == 'kostas':
        abort(400)

    # TODO delete table

    return '', 204


app.run()
