import algorithms
import json
import asyncio
import time

current_time = lambda: int(round(time.time() * 1000))


class Task:
    def __init__(self, db_objects, table_id, module, params):
        self.localtable = "local" + table_id
        self.globaltable = "global" + table_id
        self.viewlocaltable = "localview" + table_id
        self.globalresulttable = "globalres" + table_id
        self.algorithm = module
        self.params = params
        self.attributes = params['attributes']
        self.parameters = params['parameters']
        self.db_objects = db_objects


    def bindparameters(self, parameters):
        boundparam = []
        for i in parameters:
            if isinstance(i, (int, float, complex)):
                bindparam.append(i)
            else:
                boudparam.append(self.db_objects["global"]["async_con"].bind_str(i))
        return boundparam


    async def local_execute(self, local,  id, sqlscript):
        query = (
            "delete from "
            + self.localtable
            + "_"
            + str(id)
            + "; insert into "
            + self.localtable
            + "_"
            + str(id)
            + " "
            + sqlscript
        )
        await local.cursor().execute(query)


    async def create_view(self, local, attributes, table, filterpart, vals):
        if filterpart == " ":
            query = (
                "CREATE VIEW "
                + self.viewlocaltable
                + " AS select "
                + ",".join(attributes)
                + " from "
                + table
                + ";"
            )
            await local.cursor().execute(query)
        else:
            query = (
                "CREATE VIEW "
                + self.viewlocaltable
                + " AS select "
                + ",".join(attributes)
                + " from "
                + table
                + " where "
                + filterpart
                + ";"
            )
            await local.cursor().execute(query, vals)


    async def createlocalviews(self):
        t1 = current_time()

        table = self.params["table"]
        attributes = []
        for attribute in self.params["attributes"]:
            attributes.append(attribute)
        for formula in self.params["filters"]:
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
        for j, formula in enumerate(self.params["filters"]):
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
            if j < len(self.params["filters"]) - 1:
                filterpart += " or "

        create_view_calls = [
            self.create_view(
                local["async_con"],
                self.params["attributes"],
                self.params["table"],
                filterpart,
                vals,
            )
            for local in self.db_objects["local"]
        ]
        await asyncio.gather(*create_view_calls)
        print("time " + str(current_time() - t1))


    async def initialize(self, localschema,  globalschema):
        create_query = "create table if not exists %s (%s);" % (
            self.globalresulttable,
            globalschema,
        )
        await self.db_objects["global"]["async_con"].cursor().execute(
            "create table if not exists %s (%s);" % (self.globalresulttable, globalschema)
        )
        for i, local in enumerate(self.db_objects["local"]):
            query = "create table %s (%s);" % (self.localtable + "_" + str(i), localschema)
            await local["async_con"].cursor().execute(query)


    async def _local(self, iternum):
        t1 = current_time()
        sqlscript = self.algorithm._local(
            iternum, self.viewlocaltable, self.bindparameters(self.parameters), self.attributes, self.globalresulttable
        )
        local_execute_calls = [
            self.local_execute(local["async_con"], id, sqlscript)
            for id, local in enumerate(self.db_objects["local"])
        ]
        await asyncio.gather(*local_execute_calls)
        print("time " + str(current_time() - t1))


    async def _global(self, iternum):
        t1 = current_time()
        sqlscript = self.algorithm._global(
            iternum, self.globaltable, self.bindparameters(self.parameters), self.attributes
        )
        query = (
            "delete from "
            + self.globalresulttable
            + "; insert into "
            + self.globalresulttable
            + " "
            + sqlscript
        )
        await self.db_objects["global"]["async_con"].cursor().execute(query)
        cur = self.db_objects["global"]["async_con"].cursor()
        result = await cur.execute("select * from %s;" % self.globalresulttable)
        print("time " + str(current_time() - t1))
        return cur.fetchall()


    async def clean_up(self):
        await self.db_objects["global"]["async_con"].clean_tables(
            self.db_objects, self.globaltable, self.localtable, self.viewlocaltable, self.globalresulttable
        )
