from threading import Thread
import algorithms
import json
import asyncio
from monetdblib import parse_mapi_result
from pymonetdb import mapi


@asyncio.coroutine
async def local_run_inparallel(local,query):
    await local.cmd(query)
    
async def check_for_params(local,params):

    result = parse_mapi_result.parse(await local['async_con'].cmd(local['async_con'].bind("sselect id from tables where tables.system = false and tables.name = %s;",(params['table'],))))
    if result == []:
        raise Exception('Dataset does not exist in all local nodes')
    checked = []
    for attribute in params['attributes']:
               checked.append(attribute)
               attr = parse_mapi_result.parse(await local['async_con'].cmd(local['async_con'].bind("sselect name from columns where table_id = '"+str(result[0][0])+"' and name = %s;",(attribute,))));
               if attr == []:
                   raise Exception('Attribute '+attribute+' does not exist in all local nodes')
           
    for formula in params['filters']:
             for attribute in formula:
               if attribute not in checked:
                   checked.append(attribute)
                   attr = parse_mapi_result.parse(await local['async_con'].cmd(local['async_con'].bind("sselect name from columns where table_id = '"+str(result[0][0])+"' and name = %s;",(attribute[0],))));
                   if attr == []:
                       raise Exception('Attribute '+attribute[0]+' does not exist in all local nodes')


def create_view_parallel(local,query):
    local.cmd(query)
   
async def createlocalviews(db_objects, viewlocaltable, params):
      
      await asyncio.gather(*[check_for_params(local,params) for i,local in enumerate(db_objects['local'])] )


      filterpart = " "
      vals = []
      for j,formula in enumerate(params["filters"]):
        andpart = " "
        for i,filt in enumerate(formula):
          if filt[1] not in [">","<","<>",">=","<=","="]:
              raise Exception('Operator '+filt[1]+' not valid')
          andpart += filt[0] + filt[1] + "%s"
          vals.append(filt[2]) 
          if i < len(formula)-1:
              andpart += ' and '
        if andpart!=" ":
            filterpart += "("+andpart+")"
        if j < len(params["filters"])-1:
              filterpart += ' or '
              
      threads = []
      for i,local in enumerate(db_objects['local']):
        if filterpart == " ":
          t = Thread(target = create_view_parallel, args = (local['con'],"sCREATE VIEW "+viewlocaltable+" AS select "+','.join(params['attributes'])+" from "+params['table']+";"))
          t.start()
          threads.append(t)  
        else:
          t = Thread(target = create_view_parallel, args = (local['con'], local['async_con'].bind("sCREATE VIEW "+viewlocaltable+" AS select "+','.join(params['attributes'])+" from "+params['table']+" where"+ filterpart +";", vals)))
          t.start()
          threads.append(t)  
      for t in threads:
          t.join()


      

async def run_local_init(db_objects,localtable, algorithm, parameters, attr, viewlocaltable, localschema):
      for i,local in enumerate(db_objects['local']):
           local['con'].cmd("screate table %s (%s);" %(localtable+"_"+str(i),localschema))
      await asyncio.gather(*[local_run_inparallel(local['async_con'],"sinsert into "+localtable+"_"+str(i)+" "+algorithm._local_init(viewlocaltable, parameters, attr)) for i,local in enumerate(db_objects['local'])] )
      
async def run_local(db_objects,localtable, algorithm, parameters, attr, viewlocaltable, localschema):
       for i,local in enumerate(db_objects['local']):
           local['con'].cmd("screate table %s (%s);" %(localtable+"_"+str(i),localschema))
       await asyncio.gather(*[local_run_inparallel(local['async_con'],"sinsert into "+localtable+"_"+str(i)+" "+algorithm._local(viewlocaltable, parameters, attr)) for i,local in enumerate(db_objects['local'])] )

async def run_local_iter(db_objects,localtable,globalresulttable, algorithm, parameters, attr, viewlocaltable, localschema):
      for i,local in enumerate(db_objects['local']):
           local['con'].cmd("screate table %s (%s);" %(localtable+"_"+str(i),localschema))
      await asyncio.gather(*[local_run_inparallel(local['async_con'],"sinsert into "+localtable+"_"+str(i)+" "+algorithm._local_iter(globalresulttable, parameters, attr)) for i,local in enumerate(db_objects['local'])] )
      
async def run_global_final(db_objects, globaltable, algorithm, parameters, attr):
      result = await db_objects['global']['async_con'].cmd("s"+algorithm._global(globaltable, parameters, attr))
      return parse_mapi_result.parse(result)
      
async def run_global_iter(db_objects, globaltable, localtable, globalresulttable, algorithm, parameters, attr, viewlocaltable, globalschema):
      db_objects['global']['con'].cmd("sdrop table if exists %s;" %globalresulttable)
      db_objects['global']['con'].cmd("screate table %s (%s);" %(globalresulttable,globalschema))
      await db_objects['global']['async_con'].cmd("sinsert into " + globalresulttable + " " +algorithm._global_iter(globaltable, parameters, attr))
      await iteration_clean_up(db_objects, globaltable, localtable, viewlocaltable)

async def iteration_clean_up(db_objects, globaltable, localtable, viewlocaltable):
      await asyncio.sleep(0)
      db_objects['global']['con'].cmd("sdrop table if exists %s;" %globaltable)
      for i,local in enumerate(db_objects['local']):
          local['con'].cmd("sdrop table if exists "+localtable+"_"+str(i)+";")
          db_objects['global']['con'].cmd("sdrop table if exists "+localtable+"_"+str(i)+";")

async def clean_up(db_objects, globaltable, localtable, viewlocaltable, globalrestable):
      await asyncio.sleep(0)
      db_objects['global']['con'].cmd("sdrop table if exists %s;" %globaltable)
      db_objects['global']['con'].cmd("sdrop table if exists %s;" %globalrestable)
      for i,local in enumerate(db_objects['local']):
          local['con'].cmd("sdrop view if exists "+viewlocaltable+";")
          local['con'].cmd("sdrop table if exists "+globalrestable+";")
          local['con'].cmd("sdrop table if exists "+localtable+"_"+str(i)+";")
          db_objects['global']['con'].cmd("sdrop table if exists "+localtable+"_"+str(i)+";")