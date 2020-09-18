import asyncio

async def broadcast_inparallel(local, globalresulttable, globalschema, dbname ):
        await local.cursor().execute("DROP TABLE IF EXISTS %s;" %globalresulttable)
        await local.cursor().execute("CREATE REMOTE TABLE %s (%s) on 'mapi:%s';" %(globalresulttable, globalschema, dbname))

async def merge(db_objects, localtable, globaltable, localschema):
    con = db_objects['global']['async_con']
    await con.cursor().execute("DROP TABLE IF EXISTS %s;" %globaltable);
    await con.cursor().execute("CREATE MERGE TABLE %s (%s);" %(globaltable,localschema));
    for i,local_node in enumerate(db_objects['local']):
        await con.cursor().execute("DROP TABLE IF EXISTS %s_%s;" %(localtable, i))
        print("CREATE REMOTE TABLE %s_%s (%s) on 'mapi:%s';" %(localtable, i, localschema,local_node['dbname']))
        await con.cursor().execute("CREATE REMOTE TABLE %s_%s (%s) on 'mapi:%s'; " %(localtable, i, localschema,local_node['dbname']))
        await con.cursor().execute("ALTER TABLE %s ADD TABLE %s_%s;" %(globaltable,localtable,i));
    
    
async def broadcast(db_objects, globalresulttable, globalschema):
    await asyncio.gather(*[broadcast_inparallel(local_node['async_con'], globalresulttable, globalschema, db_objects['global']['dbname']) for i,local_node in enumerate(db_objects['local'])])


async def transferdirect(node1, localtable, node2, transferschema):
    await node2[2].cursor().execute("DROP TABLE IF EXISTS %s;" %localtable)
    await node2[2].cursor().execute("CREATE REMOTE TABLE %s (%s) on 'mapi:%s';" %(localtable, transferschema,node1[1]))
        
def transferviaglobal(node1, globalnode, node2, localtable):
    #same as above but not direct between the 2 local nodes
    pass