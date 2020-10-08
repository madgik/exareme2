import asyncio
from urllib.parse import urlparse
import servers
from algorithms import udfs


class Connections:
    def __init__(self):
        self.db_objects = {}
        self.db_objects["local"] = []
        self.db_objects["global"] = {}
        self.mservers = []
        self.glob = urlparse(servers.servers[0])
        if self.glob.scheme == "monetdb":
            from aiopymonetdb import pool

            self.user = "monetdb"
            self.password = "monetdb"
        if self.glob.scheme == "postgres":
            from aiopg import pool

            self.user = "postgres"
            self.password = "mypassword"
        self.pool = pool
        self.lock = asyncio.Lock()

    async def initialize(self):  ### create connection pools
        if self.db_objects["global"] == {}:
            self.mservers = servers.servers
            glob = self.glob
            self.db_objects["global"]["pool"] = await self.pool.create_pool(
                host=glob.hostname,
                port=glob.port,
                user=self.user,
                password=self.password,
                database=glob.path[1:],
            )
            self.db_objects["global"]["dbname"] = servers.servers[
                0
            ]  ## global database name - required by remote tables to connect to the remote database

            for i, db in enumerate(servers.servers[1:]):
                loc = urlparse(db)
                pol = await self.pool.create_pool(
                    host=loc.hostname,
                    port=loc.port,
                    user=self.user,
                    password=self.password,
                    database=loc.path[1:],
                )
                local_node = {}
                local_node["pool"] = pol
                local_node["dbname"] = db
                self.db_objects["local"].append(
                    local_node
                )  # for each local node an asynchronous connection object, the database name, and a blocking connection object

            con = await self.acquire()
            await con["global"]["async_con"].init_remote_connections(con)

            for udf in udfs.udf_info.values():
                await con["global"]["async_con"].cursor().execute(udf.get("declaration"))
                for local in con["local"]:
                    await local['async_con'].cursor().execute(udf.get("declaration"))

            await self.release(con)

    async def acquire(self):  #### get connection objects
        db_conn = {}
        db_conn["local"] = []
        db_conn["global"] = {}
        await self._reload()
        conn = await self.db_objects["global"]["pool"].acquire()
        db_conn["global"][
            "async_con"
        ] = conn  ## global asynchronous connection object - this is used to execute commands on the remote database
        db_conn["global"]["dbname"] = self.db_objects["global"][
            "dbname"
        ]  ## global database name - required by remote tables to connect to the remote database

        for db_object in self.db_objects["local"]:
            loc = urlparse(db_object["dbname"])
            conn = await db_object["pool"].acquire()
            local_node = {}
            local_node["async_con"] = conn
            local_node["dbname"] = db_object["dbname"]
            db_conn["local"].append(
                local_node
            )  # for each local node an asynchronous connection object, the database name
        return db_conn

    ### asyncio locks because there may be a reload
    async def release(self, db_conn):  ### release connection objects back to pool
        await self.lock.acquire()
        if db_conn["global"]["dbname"] == self.db_objects["global"]["dbname"]:
            await self.db_objects["global"]["pool"].release(
                db_conn["global"]["async_con"]
            )
        self.lock.release()
        for i, local in enumerate(self.db_objects["local"]):
            await self.lock.acquire()
            if db_conn["local"][i]["dbname"] == local["dbname"]:
                await local["pool"].release(db_conn["local"][i]["async_con"])
            self.lock.release()

    ###### reload federation nodes
    async def _reload(self):
        import importlib

        importlib.reload(servers)
        if self.mservers != servers.servers:
            await self.clearall()  #### re-init all the connections
            self.__init__()
            await self.initialize()
            ##### commented section - solve only the updates in the servers.
            # if (self.mservers[0] != servers.servers[0]):
            #    await self._update_global(servers.servers[0])
            # else:
            #    await self._update_local(self.serversdiff(servers.servers[1:],self.mservers[1:]), self.serversdiff(self.mservers[1:],servers.servers[1:]))
            # self.mservers = servers.servers
            # return 1
        return 0

    async def clearall(self):
        await self.db_objects["global"]["pool"].clear()
        for i in self.db_objects["local"]:
            await i["pool"].clear()