import algorithms
import json
import asyncio

import time

current_time = lambda: int(round(time.time() * 1000))


@asyncio.coroutine
async def local_run_inparallel(local,query):
    await local.cursor().execute(query)


async def create_view_parallel(local,query,params = None):
    await local.cursor().execute(query,params)
   
async def createlocalviews(db_objects, viewlocaltable, params):
      t1 = current_time()
      t = params['table']
      attributes = []
      for attribute in params['attributes']:
          attributes.append(attribute)
      for formula in params['filters']:
          for attribute in formula:
              attributes.append(attribute)
      await asyncio.gather(*[local['async_con'].check_for_params(t,attributes) for i,local in enumerate(db_objects['local'])] )

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

      if filterpart == " ":
          await asyncio.gather(*[create_view_parallel(local['async_con'],"CREATE VIEW "+viewlocaltable+" AS select "+','.join(params['attributes'])+" from "+params['table']+";") for i, local in enumerate(db_objects['local'])])
      else:
          await asyncio.gather(*[create_view_parallel(local['async_con'],"CREATE VIEW " + viewlocaltable + " AS select " + ','.join(params['attributes']) + " from " + params['table'] + " where"+ filterpart +";", vals) for i, local in enumerate(db_objects['local'])])
      print("time "+str(current_time()-t1))


async def _init(db_objects,localtable,localschema,globalresulttable,globalschema):
      await db_objects['global']['async_con'].cursor().execute("create table if not exists %s (%s);" % (globalresulttable, globalschema))
      for i,local in enumerate(db_objects['local']):
           await local['async_con'].cursor().execute("create table %s (%s);" %(localtable+"_"+str(i),localschema))

async def _local(iternum, globalresulttable, parameters, attr, db_objects,localtable, algorithm, viewlocaltable, localschema):
      t1 = current_time()
      await asyncio.gather(*[local_run_inparallel(local['async_con'],"delete from "+localtable + "_" + str(i)+"; insert into "+localtable+"_"+str(i)+" "+algorithm._local(iternum,viewlocaltable, parameters, attr, globalresulttable) ) for i,local in enumerate(db_objects['local'])] )
      print("time " + str(current_time() - t1))

async def _global(iternum, globaltable, parameters, attr, db_objects, localtable, globalresulttable, algorithm, viewlocaltable, globalschema):
    t1 = current_time()
    await db_objects['global']['async_con'].cursor().execute("delete from " + globalresulttable + "; insert into " + globalresulttable + " " + algorithm._global(iternum, globaltable, parameters, attr))
    cur = db_objects['global']['async_con'].cursor()
    result = await cur.execute("select * from %s;" % globalresulttable)
    print("time " + str(current_time() - t1))
    return cur.fetchall()


async def clean_up(db_objects, globaltable, localtable, viewlocaltable, globalrestable):
      await db_objects['global']['async_con'].clean_tables(db_objects, globaltable, localtable, viewlocaltable, globalrestable)
