import datetime
import random
import task
import transfer
import json
import importlib
import parser
import symbol, token
import inspect
import re

def get_package(algorithm):
    mpackage = "algorithms"
    importlib.import_module(mpackage)
    algo = importlib.import_module("."+algorithm,mpackage)
    return algo 

def get_uniquetablename():
      return 'user{0}'.format(datetime.datetime.now().microsecond + (random.randrange(1, 100+1) * 100000))

######### algorithm params:
# algorithm: the algorithm module
# db_objects: the connection objects and the database names
# localtable: the name of the result table in localnodes
# globaltable: the name of the table in global node which contains the merge of all the local result tables
# viewlocaltable: the view which contains the data that will be processed
# globalresulttable: the name of the result table in globalnode, not used here since simple local global algorithms just return their global result without storing it.
# localschema: the schema of the result table in localnodes
# globalschema: the schema of the result table in global node

async def dataflow(algorithm, parameters, attr, db_objects, localtable, globaltable,  viewlocaltable, localschema, globalresulttable = None, globalschema = None):
    await task._init(db_objects,localtable,localschema,globalresulttable,globalschema)

    src = inspect.getsource(algorithm.dataflow)
    func = [0]

    ###### of course this MUST change and use Python's parser module to edit bytecode ########################
    src = re.sub("\_global\(iternum(?:\s)*\,(?:\s)*globaltable(?:\s)*\,(?:\s)*parameters(?:\s)*\,(?:\s)*attr(?:\s)*\)","await task._global(iternum, globaltable, parameters, attr, db_objects, localtable, globalresulttable, algorithm, viewlocaltable, globalschema)",src)
    src = re.sub("\_local\(iternum(?:\s)*\,(?:\s)*viewlocaltable(?:\s)*\,(?:\s)*parameters(?:\s)*\,(?:\s)*attr(?:\s)*,(?:\s)*globalresulttable\)","await task._local(iternum, globalresulttable, parameters, attr, db_objects,localtable, algorithm, viewlocaltable, localschema)",src)
    src = src.split('\n', 1)[-1]
    src = "async def dataflow(algorithm, parameters, attr, db_objects, localtable, globaltable,  viewlocaltable, localschema, globalresulttable = None, globalschema = None):\n" + src + "\nfunc[0] = dataflow"
    #########################################################################################

    ast = parser.suite(src)
    exec(parser.compilest(ast))
    return await func[0](algorithm, parameters, attr, db_objects, localtable, globaltable,  viewlocaltable, localschema, globalresulttable, globalschema)



#### run function:
# get the algorithm name and accesses the corresponding python module
# parses and processes the params which contain: the table name, the attributes and the filters
# gets also db_objects: the connection objects to all the nodes of the federation
# it decides the type of the algorithm using schema.json file, calls the appropriate execution function
# cleans up the servers and returns the results

     
async def run(algorithm, params, db_objects):
      #### create unique table names
      table_id = get_uniquetablename()
      localtable = "local"+table_id
      globaltable = "global"+table_id
      viewlocaltable = 'localview'+table_id
      globalresulttable = "globalres"+table_id
      result = []
      params = json.loads(params)
      parameters = params['parameters']
      
      #### bind string algorithm parameters, to avoid sql injections
      bindparams = []
      for i in parameters:
         if isinstance(i, (int, float, complex)):
             bindparams.append(i)
         else:
             bindparams.append(db_objects['global']['async_con'].bind_str(i))
          
      ### get the corresponding algorithm python module using algorithm name
      module = get_package(algorithm)
      # create database views on local databases - each view processes the filters and the selected attributes on the requested table
      # the algorithm won't run directly on the local dataset but on the view
      await task.createlocalviews(db_objects, viewlocaltable, params)
      
      ##### schema.json contains info about each algorithm: the name, the type (simple, iterative etc.) and the intermediate result schema
      with open('schema.json') as json_file:
          data = json.load(json_file)

      for c,algo in enumerate([ data['algorithms'][i]['name'] for i,j in enumerate(data['algorithms'])]):
          if algorithm == algo:
            try:
                await transfer.initialize(db_objects, localtable, globaltable, globalresulttable, data['algorithms'][c]['local_schema'],data['algorithms'][c]['global_schema'])
                result =  await dataflow(module,bindparams, params['attributes'], db_objects, localtable, globaltable, viewlocaltable, data['algorithms'][c]['local_schema'], globalresulttable, data['algorithms'][c]['global_schema'])
            except:
                 await task.clean_up(db_objects, globaltable, localtable, viewlocaltable, globalresulttable)
                 raise
      ### clean up tables that are created during the execution
      await task.clean_up(db_objects, globaltable, localtable, viewlocaltable, globalresulttable)
      return result
