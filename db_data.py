import asyncio
import datetime
import random

from algorithms.udfs import udf_info


class Data:
    async def initialize_from_table(self, db_objects, input_table_name: str, input_table_schema: str):
        self.db_objects = db_objects
        self.table_name = input_table_name
        self.table_schema = input_table_schema

    async def run_udf(self, name):
        pass

    def get_value(self):
        pass


class LocalData(Data):
    def __init__(self):
        self.unique_id = get_unique_id()
        self.table_name = None
        self.table_schema = None
        self.db_objects = None

    async def initialize_from_params(self, params, db_objects):
        self.db_objects = db_objects
        self.table_name = "local_view_" + self.unique_id
        self.table_schema = None  # TODO Initialize with schema of view if possible
        await self.create_local_views_from_params(params)

    async def create_local_views_from_params(self, params):
        table = params["table"]
        attributes = []
        for attribute in params["attributes"]:
            attributes.append(attribute)
        for formula in params["filters"]:
            for attribute in formula:
                if attribute[0] not in attributes:
                    attributes.append(attribute[0])
        check_for_params_calls = [
            local["async_con"].check_for_params(table, attributes)
            for i, local in enumerate(self.db_objects["local"])
        ]
        await asyncio.gather(*check_for_params_calls)

        filterpart = " "
        vals = []
        for j, formula in enumerate(params["filters"]):
            andpart = " "
            for i, filt in enumerate(formula):
                if filt[1] not in [">", "<", "<>", ">=", "<=", "="]:
                    raise Exception("Operator " + filt[1] + " not valid")
                andpart += filt[0] + filt[1] + "%s"
                vals.append(filt[2])
                if i < len(formula) - 1:
                    andpart += " and "
            if andpart != " ":
                filterpart += "(" + andpart + ")"
            if j < len(params["filters"]) - 1:
                filterpart += " or "

        create_view_calls = [
            self.create_local_view_from_params(
                local["async_con"],
                params["attributes"],
                get_node_specific_name(self.table_name, unique_node_identifier),
                params["table"],
                filterpart,
                vals,
            )
            for unique_node_identifier, local in enumerate(self.db_objects["local"])
        ]
        await asyncio.gather(*create_view_calls)

    async def create_local_view_from_params(self, db_connection, attributes, view_name, table, filterpart, vals):
        if filterpart == " ":
            query = (
                    "CREATE VIEW "
                    + view_name
                    + " AS select "
                    + ",".join(attributes)
                    + " from "
                    + table
                    + ";"
            )
            await db_connection.cursor().execute(query)
        else:
            query = (
                    "CREATE VIEW "
                    + view_name
                    + " AS select "
                    + ",".join(attributes)
                    + " from "
                    + table
                    + " where "
                    + filterpart
                    + ";"
            )
            await db_connection.cursor().execute(query, vals)

    async def run_udf(self, name) -> Data:

        # TODO Check if udf exists

        if name.startswith('local_'):
            return await self.run_local_udf(name)
        elif name.startswith('global_'):
            return await self.run_global_udf(name)
        else:
            raise Exception('Invalid udf name. Udfs should start with "local_" or "global_".')

    async def run_local_udf(self, udf_name: str) -> 'LocalData':

        output_table_name = "LOCAL_TABLE_" + get_unique_id()
        output_table_schema = udf_info[udf_name]["return_schema"]

        await self.create_local_tables(output_table_name, output_table_schema)

        execution_of_udf = [
            run_udf_on_node(
                db_connection["async_con"],
                udf_name,
                get_node_specific_name(self.table_name, unique_node_identifier),
                get_node_specific_name(output_table_name, unique_node_identifier)
            ) for unique_node_identifier, db_connection in enumerate(self.db_objects["local"])
        ]
        await asyncio.gather(*execution_of_udf)

        new_local_data = LocalData()
        await new_local_data.initialize_from_table(self.db_objects, output_table_name, output_table_schema)
        return new_local_data

    async def create_local_tables(self, table_name: str, table_schema: str):

        execution_of_create_tables = [
            create_table(local_db_connection["async_con"],
                         get_node_specific_name(table_name, unique_node_identifier),
                         table_schema
                         )
            for unique_node_identifier, local_db_connection in enumerate(self.db_objects["local"])
        ]
        await asyncio.gather(*execution_of_create_tables)

    async def run_global_udf(self, udf_name: str) -> 'GlobalData':

        united_global_table_name = await self.unite_local_tables()

        output_table_name = "GLOBAL_TABLE_" + get_unique_id()
        output_table_schema = udf_info[udf_name]["return_schema"]

        await create_table(
            self.db_objects["global"]["async_con"],
            output_table_name,
            output_table_schema
        )

        await run_udf_on_node(
            self.db_objects["global"]["async_con"],
            udf_name,
            united_global_table_name,
            output_table_name
        )

        new_global_data = GlobalData()
        await new_global_data.initialize_from_table(self.db_objects, output_table_name, output_table_schema)
        return new_global_data

    async def unite_local_tables(self) -> str:

        united_global_table_name = "UNITED_GLOBAL_TABLE_" + self.unique_id

        await create_merge_table(self.db_objects["global"]["async_con"], united_global_table_name, self.table_schema)

        # TODO Async?
        for unique_node_identifier, db_info in enumerate(self.db_objects["local"]):
            await create_remote_table(
                self.db_objects["global"]["async_con"],
                get_node_specific_name(self.table_name, unique_node_identifier),
                self.table_schema,
                db_info["dbname"]
            )

        for unique_node_identifier, db_info in enumerate(self.db_objects["local"]):
            await add_table_to_merge_table(
                self.db_objects["global"]["async_con"],
                get_node_specific_name(self.table_name, unique_node_identifier),
                united_global_table_name
            )

        return united_global_table_name

    def get_value(self):
        raise NotImplementedError("Cannot get the value from Local data.")


class GlobalData(Data):
    def __init__(self):
        print('1')

    async def get_value(self):
        db_cursor= self.db_objects["global"]["async_con"].cursor()
        await db_cursor.execute("SELECT * FROM " + self.table_name + ";")
        return db_cursor.fetchall()


async def run_udf_on_node(db_connection, udf_name: str, input_table_name: str,
                          output_table_name: str):
    # TODO Check if UDF can run on these data
    udf_sql_query: str = "select * from " + udf_name + "((select * from " + input_table_name + "));"

    query = (
            "INSERT INTO " + output_table_name + " "
            + udf_sql_query
    )
    await db_connection.cursor().execute(query)


async def create_table(db_connection, table_name: str, table_schema: str):
    query = (
            "CREATE TABLE " + table_name + " "
            + "(" + table_schema + ");"
    )
    await db_connection.cursor().execute(query)


async def create_merge_table(db_connection, table_name: str, table_schema: str):
    query = (
            "CREATE MERGE TABLE " + table_name + " "
            + "(" + table_schema + ");"
    )
    await db_connection.cursor().execute(query)


async def create_remote_table(db_connection, table_name, table_schema, remote_db_name):
    remote_table_query = (
            "CREATE REMOTE TABLE " + table_name + " (" + table_schema + ") "
            + "on 'mapi:" + remote_db_name + "';"
    )
    await db_connection.cursor().execute(remote_table_query)


async def add_table_to_merge_table(db_connection, table_name, merge_table_name, ):
    update_merge_table_query = (
            "ALTER TABLE " + merge_table_name + " ADD TABLE " + table_name + ";"
    )
    await db_connection.cursor().execute(update_merge_table_query)


def get_node_specific_name(name: str, unique_node_identifier: int) -> str:
    return name + "_" + str(unique_node_identifier)


def get_unique_id():
    return str(datetime.datetime.now().microsecond + (random.randrange(1, 100 + 1) * 100000))
