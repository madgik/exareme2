import merge_broadcast
import asyncio

async def merge(db_objects, localtable, globaltable, localschema):
    await asyncio.sleep(0)
    merge_broadcast.merge(db_objects, localtable, globaltable, localschema)

async def broadcast(db_objects, globalresulttable, globalschema):
    await asyncio.sleep(0)
    merge_broadcast.broadcast(db_objects, globalresulttable, globalschema)
    
async def transfer(node1,  localtable, node2, transferschema):
    await asyncio.sleep(0)
    merge_broadcast.transferdirect(node1, localtable, node2)

