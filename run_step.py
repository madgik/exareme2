from threading import Thread
import algorithms
import json
import asyncio
from pymonetdb import mapi

import time

current_time = lambda: int(round(time.time() * 1000))


@asyncio.coroutine
async def local_run_inparallel(local,query):
    await local.cursor().execute(query)
    
async def check_for_params(local,params):
    cur = local['async_con'].cursor()
    result = await cur.execute("select id from tables where tables.system = false and tables.name = %s;",(params['table'],))

    if result == 0:
        raise Exception('Dataset does not exist in all local nodes')
    checked = []
    table_id = cur.fetchone()[0]
    for attribute in params['attributes']:
               checked.append(attribute)
               attr = await local['async_con'].cursor().execute("select name from columns where table_id = '"+str(table_id)+"' and name = %s;",(attribute,))
               if attr == 0:
                   raise Exception('Attribute '+attribute+' does not exist in all local nodes')
           
    for formula in params['filters']:
             for attribute in formula:
               if attribute not in checked:
                   checked.append(attribute)
                   attr = await local['async_con'].cursor().execute("select name from columns where table_id = '"+str(table_id)+"' and name = %s;",(attribute[0],))
                   if attr == 0:
                       raise Exception('Attribute '+attribute[0]+' does not exist in all local nodes')


async def create_view_parallel(local,query,params = None):
    await local.cursor().execute(query,params)
   
async def createlocalviews(db_objects, viewlocaltable, params):
      t1 = current_time()
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
              


      if filterpart == " ":
          await asyncio.gather(*[create_view_parallel(local['async_con'],"CREATE VIEW "+viewlocaltable+" AS select "+','.join(params['attributes'])+" from "+params['table']+";") for i, local in enumerate(db_objects['local'])])
      else:
          await asyncio.gather(*[create_view_parallel(local['async_con'],"CREATE VIEW " + viewlocaltable + " AS select " + ','.join(params['attributes']) + " from " + params['table'] + " where"+ filterpart +";", vals) for i, local in enumerate(db_objects['local'])])
      print("time "+str(current_time()-t1))

async def run_local_init(db_objects,localtable, algorithm, parameters, attr, viewlocaltable, localschema, globalschema, globalresulttable):
      t1 = current_time()
      await db_objects['global']['async_con'].cursor().execute("create table if not exists %s (%s);" % (globalresulttable, globalschema))
      for i,local in enumerate(db_objects['local']):
           await local['async_con'].cursor().execute("create table %s (%s);" %(localtable+"_"+str(i),localschema))
      await asyncio.gather(*[local_run_inparallel(local['async_con'],"insert into "+localtable+"_"+str(i)+" "+algorithm._local_init(viewlocaltable, parameters, attr)) for i,local in enumerate(db_objects['local'])] )
      print("time " + str(current_time() - t1))

async def run_local(db_objects,localtable, algorithm, parameters, attr, viewlocaltable, localschema):
       for i,local in enumerate(db_objects['local']):
           await local['async_con'].cursor().execute("create table %s (%s);" %(localtable+"_"+str(i),localschema))
       await asyncio.gather(*[local_run_inparallel(local['async_con'],"insert into "+localtable+"_"+str(i)+" "+algorithm._local(viewlocaltable, parameters, attr)) for i,local in enumerate(db_objects['local'])] )


async def run_local_iter(db_objects,localtable,globalresulttable, algorithm, parameters, attr, viewlocaltable, localschema):
      t1 = current_time()
      for i,local in enumerate(db_objects['local']):
          await local['async_con'].cursor().execute("delete from %s;" % (localtable + "_" + str(i),))

      await asyncio.gather(*[local_run_inparallel(local['async_con'],"insert into "+localtable+"_"+str(i)+" "+algorithm._local_iter(globalresulttable, parameters, attr)) for i,local in enumerate(db_objects['local'])] )
      print("time " + str(current_time() - t1))

async def run_global_final(db_objects, globaltable, algorithm, parameters, attr):
      t1 = current_time()
      cur = db_objects['global']['async_con'].cursor()
      result = await cur.execute(algorithm._global(globaltable, parameters, attr))
      print("time " + str(current_time() - t1))
      return cur.fetchall()

      
async def run_global_iter(db_objects, globaltable, localtable, globalresulttable, algorithm, parameters, attr, viewlocaltable, globalschema):
      t1 = current_time()
      await db_objects['global']['async_con'].cursor().execute("delete from %s;" % (globalresulttable,))
      await db_objects['global']['async_con'].cursor().execute("insert into " + globalresulttable + " " +algorithm._global_iter(globaltable, parameters, attr))
      print("time " + str(current_time() - t1))


async def clean_up(db_objects, globaltable, localtable, viewlocaltable, globalrestable):
      await db_objects['global']['async_con'].cursor().execute("drop table if exists %s;" %globaltable)
      await db_objects['global']['async_con'].cursor().execute("drop table if exists %s;" %globalrestable)
      for i,local in enumerate(db_objects['local']):
          await local['async_con'].cursor().execute("drop view if exists "+viewlocaltable+";")
          await local['async_con'].cursor().execute("drop table if exists "+globalrestable+";")
          await local['async_con'].cursor().execute("drop table if exists "+localtable+"_"+str(i)+";")
          await db_objects['global']['async_con'].cursor().execute("drop table if exists "+localtable+"_"+str(i)+";")