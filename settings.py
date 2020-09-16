from monetdblib import mapi_async
from pymonetdb import mapi
import asyncio
from urllib.parse import urlparse
from monetdblib import pool
import servers


class Settings:

    def __init__(self):
        self.db_objects = {}
        self.db_objects['local'] = []
        self.db_objects['global'] = {}
        self.mservers = []
        
    async def initialize(self):  ### create connection pools
        if self.db_objects['global'] == {}:
          self.mservers = servers.servers
          glob = urlparse(servers.servers[0])
          self.db_objects['global']['pool'] = await pool.create_pool(hostname=glob.hostname, port=glob.port, username="monetdb",
            password="monetdb", database=glob.path[1:], language="sql")
            
          self.db_objects['global']['dbname'] = servers.servers[0]  ## global database name - required by remote tables to connect to the remote database
    
         
          for i,db in enumerate(servers.servers[1:]):
            loc = urlparse(db)      
       
            pol = await pool.create_pool(hostname=loc.hostname, port=loc.port, username="monetdb",
                 password="monetdb", database=loc.path[1:], language="sql")
            local_node = {}
            local_node['pool'] = pol
            local_node['dbname'] = db
            self.db_objects['local'].append(local_node)  # for each local node an asynchronous connection object, the database name, and a blocking connection object

  

    async def acquire(self): #### get connection objects 
        db_conn = {}
        db_conn['local'] = []
        db_conn['global'] = {}
        
        await self._reload()
    
        conn = await self.db_objects['global']['pool'].acquire()
    
        db_conn['global']['async_con'] = conn   ## global asynchronous connection object - this is used to execute commands on the remote database
        db_conn['global']['dbname'] = self.db_objects['global']['dbname']  ## global database name - required by remote tables to connect to the remote database
    
        glob = urlparse(self.db_objects['global']['dbname'])
        server = mapi.Connection()
        server.connect(hostname=glob.hostname, port=glob.port, username="monetdb",
                 password="monetdb", database=glob.path[1:], language="sql")
    
        db_conn['global']['con'] = server ## global blocking connection objects - required at this time because monetdb does not support create commands concurrently
     
        for db_object in self.db_objects['local']:
            loc = urlparse(db_object['dbname'])      
            conn = await db_object['pool'].acquire()
            server2 = mapi.Connection()
            server2.connect(hostname=loc.hostname, port=loc.port, username="monetdb",
                 password="monetdb", database=loc.path[1:], language="sql")
        
            local_node = {}
            local_node['async_con'] = conn
            local_node['dbname'] = db_object['dbname']
            local_node['con'] = server2
            db_conn['local'].append(local_node)  # for each local node an asynchronous connection object, the database name, and a blocking connection object
        return db_conn


     ### asyncio locks because there may be a reload 
    async def release(self,db_conn): ### release connection objects back to pool
        lock = asyncio.Lock()
        await lock.acquire()
        if (db_conn['global']['dbname'] == self.db_objects['global']['dbname']):
            self.db_objects['global']['pool']._release(db_conn['global']['async_con'])
        db_conn['global']['con'].disconnect()
        lock.release()
        for i,local in enumerate(self.db_objects['local']):
            await lock.acquire()
            if (db_conn['local'][i]['dbname'] == local['dbname']):
                local['pool']._release(db_conn['local'][i]['async_con'])
                db_conn['local'][i]['con'].disconnect()
            lock.release()

    async def _update_global(self,server):  #### update global server if servers file is reloaded
        await self.db_objects['global']['pool'].clear()
        glob = urlparse(server)
        dbpool['global'] = await pool.create_pool(hostname=glob.hostname, port=glob.port, username="monetdb",
            password="monetdb", database=glob.path[1:], language="sql")


    async def _update_local(self,added, removed): #### update local servers if servers file is reloaded
        for local in removed:
            c = 0
            for i in self.db_objects['local']:
                  
                  if i['dbname'] == local:
                       await i['pool'].clear()
                       del self.db_objects['local'][c]
                       break
                  c+=1
           # await self.db_objects['local'][c]['pool'].clear()
            
         
        for local in added:
            loc = urlparse(local)      
            pol = await pool.create_pool(hostname=loc.hostname, port=loc.port, username="monetdb",
                 password="monetdb", database=loc.path[1:], language="sql")
            local_node = {}
            local_node['pool'] = pol
            local_node['dbname'] = local
            self.db_objects['local'].append(local_node)

        
    #### these two functions calculate the updates to the servers files (new nodes and deleted nodes)
    def old_minus_new(self,first, second):
        second = set(second)
        return [item for item in first if item not in second]

    def new_minus_old(self,first, second):
        second = set(second)
        return [item for item in first if item not in second]

    ###### reload federation nodes
    async def _reload(self):
        import importlib
        importlib.reload(servers)
        if self.mservers != servers.servers:
            if (self.mservers[0] != servers.servers[0]):
                await self._update_global(servers.servers[0])
            else:
                await self._update_local(self.new_minus_old(servers.servers[1:],self.mservers[1:]), self.old_minus_new(self.mservers[1:],servers.servers[1:]))
            self.mservers = servers.servers
            return 1
        return 0
    
   


    async def clearall(self):
        await dbpool['global'].clear()
        for i in dbpool['local']:
            await i.clear()

#    await  db_objects['global']['async_con'].disconnect()
#    db_objects['global']['con'].disconnect()
#    for local in db_objects['local']:
#        await local['async_con'].disconnect()
#        local['con'].disconnect()
#    db_objects = {}
    



