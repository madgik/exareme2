import init_transfer_tables
import asyncio

async def initialize(db_objects, localtable, globaltable, globalresulttable, localschema, globalschema  =  None):
    await merge(db_objects, localtable, globaltable, localschema)
    if globalschema:
        await broadcast(db_objects, globalresulttable, globalschema)

async def merge(db_objects, localtable, globaltable, localschema):
    await init_transfer_tables.merge(db_objects, localtable, globaltable, localschema)

async def broadcast(db_objects, globalresulttable, globalschema):
    await init_transfer_tables.broadcast(db_objects, globalresulttable, globalschema)
    
async def transfer(node1,  localtable, node2, transferschema):
    await init_transfer_tables.transferdirect(node1, localtable, node2)

