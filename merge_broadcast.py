from threading import Thread


def broadcast_inparallel(local, globalresulttable, globalschema, dbname ):
        local.cmd("sDROP TABLE IF EXISTS %s;" %globalresulttable)
        local.cmd("sCREATE REMOTE TABLE %s (%s) on 'mapi:%s';" %(globalresulttable, globalschema, dbname))  

def merge(db_objects, localtable, globaltable, localschema):
    con = db_objects['global']['con']
    con.cmd("sDROP TABLE IF EXISTS %s;" %globaltable);
    con.cmd("sCREATE MERGE TABLE %s (%s);" %(globaltable,localschema));
    for i,local_node in enumerate(db_objects['local']):
        con.cmd("sDROP TABLE IF EXISTS %s_%s;" %(localtable, i))
        print("sCREATE REMOTE TABLE %s_%s (%s) on 'mapi:%s';" %(localtable, i, localschema,local_node['dbname']))
        con.cmd("sCREATE REMOTE TABLE %s_%s (%s) on 'mapi:%s'; " %(localtable, i, localschema,local_node['dbname']))
        con.cmd("sALTER TABLE %s ADD TABLE %s_%s;" %(globaltable,localtable,i));  
    
    
def broadcast(db_objects, globalresulttable, globalschema):
    threads = []
    for i,local_node in enumerate(db_objects['local']):
          t = Thread(target = broadcast_inparallel, args = (local_node['con'], globalresulttable, globalschema, db_objects['global']['dbname']))
          t.start()
          threads.append(t)    
    for t in threads:
          t.join()

    
def transferdirect(node1, localtable, node2, transferschema):
    node2[2].cmd("sDROP TABLE IF EXISTS %s;" %localtable)
    node2[2].cmd("sCREATE REMOTE TABLE %s (%s) on 'mapi:%s';" %(localtable, transferschema,node1[1]))
        
def transferviaglobal(node1, globalnode, node2, localtable):
    #same as above but not direct between the 2 local nodes
    pass