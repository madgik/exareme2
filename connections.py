import asyncio
from urllib.parse import urlparse
import servers
from algorithms import udfs
import importlib
import pkgutil
import algorithms
import settings
DEBUG = settings.DEBUG


def get_udfs(reload = False):
    modules = []
    for importer, modname, ispkg in pkgutil.iter_modules(algorithms.__path__):
        modules.append(modname)
## get all statically defined udfs from algorithms package
    mpackage = "algorithms"


    importlib.import_module(mpackage)
    all_udfs = []

    for algorithm in modules:
        try:
            algo = importlib.import_module("." + algorithm, mpackage)
            if reload:
                importlib.reload(algo)
            for udf in algo.udf_list:
                all_udfs.append(udf)
        except:
            pass
    return all_udfs


class Connections:
    def __init__(self):
        self.db_objects = {}
        self.db_objects["local"] = []
        self.db_objects["global"] = {}
        self.mservers = []
        self.udfs_list = []
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
            self.udfs_list = get_udfs()
            for udf in self.udfs_list:
                    try:
                        await con["global"]["async_con"].cursor().execute(udf)
                    except:
                        pass
                    for local in con["local"]:
                        try:
                            await local['async_con'].cursor().execute(udf)
                        except:
                            pass

                    # at this time due to minimal error handling and due to testing there may be tables in the DB which
                    #  are not dropped and are dependent on some UDFs, so their recreation may fail
                    # (You cannot replace a UDF which is in use)

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
        if DEBUG:
            await self._reload_udfs(db_conn)
        return db_conn

    ### TODO asyncio locks are needed because there may be a reload
    async def release(self, db_conn):  ### release connection objects back to pool
        if db_conn["global"]["dbname"] == self.db_objects["global"]["dbname"]:
            await self.db_objects["global"]["pool"].release(
                db_conn["global"]["async_con"]
            )
        for i, local in enumerate(self.db_objects["local"]):
            if db_conn["local"][i]["dbname"] == local["dbname"]:
                await local["pool"].release(db_conn["local"][i]["async_con"])

    async def _reload_udfs(self, con):
        udfs_list = get_udfs(True)

        if udfs_list != self.udfs_list:
            for udf in list(set(udfs_list) - set(self.udfs_list)):
                await con["global"]["async_con"].cursor().execute(udf)
                for local in con["local"]:
                    await local['async_con'].cursor().execute(udf)

        self.udfs_list = udfs_list

    ###### reload federation nodes
    async def _reload(self):
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


#   #### this function calculates the updates to the servers files (new nodes and deleted nodes)
#   def serversdiff(self,first, second):
#       second = set(second)
#       return [item for item in first if item not in second]

# async def _update_global(self, server):  #### update global server if servers file is reloaded
#    await
#    self.db_objects['global']['pool'].clear()
#    self.glob = urlparse(server)
#    if self.glob.scheme == 'monetdb':
#        from aiopymonetdb import pool
#        self.db_objects['global']['pool'] = await
#        pool.create_pool(host=self.glob.hostname, port=self.glob.port, user="monetdb",
#                         password="monetdb", database=self.glob.path[1:])
#    if self.glob.scheme == 'postgres':
#        from aiopg import pool
#        self.db_objects['global']['pool'] = await
#        pool.create_pool(host=self.glob.hostname, port=self.glob.port, user="postgres",
#                         password="mypostgres", database=self.glob.path[1:])
#    self.pool = pool


# async def _update_local(self, added, removed):  #### update local servers if servers file is reloaded
#    for local in removed:
#        c = 0
#        for i in self.db_objects['local']:

#            if i['dbname'] == local:
#               await
#                i['pool'].clear()
#                del self.db_objects['local'][c]
#                break
#            c += 1
# await self.db_objects['local'][c]['pool'].clear()

#    for local in added:
#        loc = urlparse(local)
#        if loc.scheme == 'monetdb':
#            from aiopymonetdb import pool
#            pol = await
#            pool.create_pool(host=loc.hostname, port=loc.port, user="monetdb",
#                             password="monetdb", database=loc.path[1:])
#        if loc.scheme == 'postgres':
#            from aiopg import pool
#            pol = await
#            pool.create_pool(host=loc.hostname, port=loc.port, user="postgres",
#                             password="mypassword", database=loc.path[1:])

#        local_node = {}
#        local_node['pool'] = pol
#        local_node['dbname'] = local
#        self.db_objects['local'].append(local_node)

#    await  db_objects['global']['async_con'].disconnect()
#    db_objects['global']['con'].disconnect()
#    for local in db_objects['local']:
#        await local['async_con'].disconnect()
#        local['con'].disconnect()
#    db_objects = {}
