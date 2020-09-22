import asyncio

async def initialize(db_objects, localtable, globaltable, globalresulttable, localschema, globalschema  =  None):
    await db_objects['global']['async_con'].merge(db_objects, localtable, globaltable, localschema)
    if globalschema:
        await db_objects['global']['async_con'].broadcast(db_objects, globalresulttable, globalschema)

async def merge(db_objects, localtable, globaltable, localschema):
    await db_objects['global']['async_con'].merge(db_objects, localtable, globaltable, localschema)

async def broadcast(db_objects, globalresulttable, globalschema):
    await db_objects['global']['async_con'].broadcast(db_objects, globalresulttable, globalschema)
    
async def transfer(node1,  localtable, node2, transferschema):
    await db_objects['global']['async_con'].transferdirect(node1, localtable, node2)

