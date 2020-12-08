# Exareme2
<b>Important note</b>
This branch is build on top of the postgres branch. It contains some clean up of the code and removal of redudant steps.
From now, all the algorithms do not have to define their type (simple, iterative) in the schema.json, and dataflow definition is moved in the [algorithm].py file. 
The way this has been done is using Python's generator coroutines. The algorithm coroutine yields the queries one after the other to the system's coroutine (dataflow) while also receives the
global results. The system (run_algorithm.py) calls the algorithm module, gets the queries runs them with the task executor's instance and sends back the intermediate results. In this way, the algorithm developer constructs the flow without having access and knowledge to system's internals, but by using the well-established python's generators to generate the flow of the queries and get the intermediate results. With using python's generators for the communication between the algorithm and the system we achieve security and solve the issue with predefined dataflows using a popular python's feature.  
Moreover, since the algorithm code does not contain any network IO there is no need the algorithm
developer to write asynchronous code with async/await. 

Το specify, the algorithm developer only yields sql queries one after another using standard Python. After a global query an extra yield is required to receive the result of the global query and continue the flow (e.g., countiter.py). This needs discussion, Lefteris disagrees with giving the result to python's algorithm developer code and prefers that the global sql query by itself defines the end of the iteration without passing result data (which can be bigger in another application) to python. This can be easily done relationally (i.e., one extra attribute in the global table which returns the result and one bool column that indicates if the iteration has finished), countiter_full_sql is an example with termination condition implemented in SQL using SQL conditional expressions. 

Currently, the system supposes that always a global step comes after a local step and vice versa. If this is not always the case, the algorithm developer could yield not only the query and the schema but also instructions/definitions about the task (if it is a local or a global task (at the time) or anyother kind of task that will be supported in the future to cover all the types of distributed dataflows). Another task could be the createudf task. In this case, the algorithm developer defines a function in python, yields the function object and then the system passes it to the task executor which registers it to the dbs. In this case, Jason's wrapper is placed in the ODBC as a connection.create_function(func) which gets a python function, wraps and registers it to the db (as already exists in other databases, duckdb, sqlite etc.) 

So, this update is also compatible with using abstractions to produce the yielded SQL scripts and wrappers to define UDFs in Python and register them to the DB automatically via the ODBC. The full implementation of an algorithm can be in one place in [algorithm].py file.
The main purpose of these generator coroutines is to isolate the algorithm from the federated system so that the algorithm developer needs only to know standard Python (to produce the flow) and standard SQL (or an abstraction, to produce the processing steps), and does not depend in any way on the system's internals and possible big changes that there may take place on these internals (e.g., concurrency support with async await) for any reason.

As for Postgres test integration, all the functionalities that are specific to the different DBMS have been moved to the connection objects of their aio libs and executed by the connection instance of the global node. This is not the perfect way to implement such an abstraction but a quick and dirty solution which is simpler at this time.
These functions contain the remote and merge tables and the cleanup (it's different to drop a monetdb remote table compared to a postgres foreign data wrapper). The other SQL functionalities (selects, create tables, create views) exist in the standard SQL and they are the same for all the DBMSes so there is no significant reason to transfer them at the time being.


<b>Installation</b>
1) Python3 with numpy should be installed in all federation nodes. The mserver runs with Python 3.7 or newer but is also compatible with JIT compiled PyPy3.  It does not run with python3 versions older than 3.7.
2) Install monetdb from source (https://www.monetdb.org/Developers/SourceCompile) to all the nodes of the federation
   dependencies: `sudo apt-get install libssl-dev` `sudo apt-get install libpcre3 libpcre3-dev` `sudo apt-get install pkg-config` `sudo apt-get install python3-dev` `sudo apt-get install uuid-dev` `sudo apt-get install libxml2 libxml2-dev` `sudo apt-get install libstreams-dev` `sudo apt install unixodbc-dev`
3) Install dependencies for mserver: `pip3 install tornado` , `pip3 install six`, `pip3 install pyodbc`
4) Create databases in each node. The tables that will take place in the federation should have the same schema in all local nodes. 
The nodes have to be really remote to play concurrently. (If all the dbs are in the same VM, strange bugs may occur).
Run the creation steps in the tmpfs of your VMs (usually in /dev/shm), since remote tables at the time are created on disk, it makes a big difference in execution times.
Detailed monetdb documentation:
https://www.monetdb.org/Documentation/UserGuide/Tutorial

<br>Example creation of a monetdb server and database: <br>

<pre><code>
monetdbd create mydbfarm
monetdbd set port=50000 mydbfarm
monetdbd set listenaddr=0.0.0.0  mydbfarm
monetdbd start mydbfarm
monetdb create voc
monetdb set embedpy3=true voc
monetdb release voc
monetdb start voc
#open client# mclient -u monetdb -d voc
pass: monetdb
</code></pre>

To stop and restart a server:

<pre><code>
monetdb stop voc
monetdbd stop mydbfarm
monetdbd start mydbfarm
monetdb start voc
</code></pre>

5) Python libraries for algorithms are in `algorithms` folder. Set this to path and update udfs.sql file that appends the path hard-coded.
6) Include in servers.py file all the global/local nodes (as in the already existing example). The first node is the global.
7) Default port for mserver is hard-coded in mserver.py file.

<b>Usage:</b> 
Run server: <br>

`python3 mserver.py`


<b>URL Request Post:</b> <br>
<br> Content-Type: application/x-www-form-urlencoded (usually the default in most libraries), type application/json is not supported in tornado web framework</br>
Two fields: <br>
<br> 1) `algorithm` (e.g., "pearson")
<br> 2) `params`: valid json including table name, attributes, parameters and filters. e.g. Filters follow the DNF (disjunctive normal form:
The innermost tuples each describe a single column predicate. The list of inner predicates is interpreted as a conjunction (AND), forming a more selective and multiple column predicate. Finally, the most outer list combines these filters as a disjunction (OR).
`{"table":"data", "attributes":["c1","c2"],"parameters":[0.7,4],"filters":[[["c1",">","2"],["c1","<","10000"]],[["c1",">","0"]]]}`
(i.e., pearson requires a table with 2 float attributes)

<br>
<b>Implement a new algorithm:</b> <br>

1) Add its UDFs to udf.py file
2) Add its lib to algorithms folder
3) Add an [algorithm name].py file to algorithms folder which returns the returned schema and the sql query 
for each step of the algorithm and defines the dataflows. 

<br>
<b>Other features:</b> <br>

1) Updating servers.py file the module is auto reloaded online and does not require restarting
2) A simple fault tolerance has been added for local nodes.

<br>

<br>Run with Postgres: <br>

1) Install Postgres in each node and create databases (setup passwords, username is `postgres` you need to set password `mypassword`).
2) Create the Postgres extension postgres_fdw https://www.postgresql.org/docs/9.5/postgres-fdw.html in all the nodes
3) You may need to install some dependencies for aiopg https://aiopg.readthedocs.io/en/stable/ https://github.com/aio-libs/aiopg 
4) `pip3 install psycopg2` is one dependency may be there are some more
5) Edit servers.py file using postgres hosts, db names etc.
6) Pearson is not implemented in postgres. It could be using postgres UDFs or naive SQL. Iterative dummy algorithm countiter is supported

<b>General comments:</b> <br>

1) Functional programming style is adopted.
2) Async non-blocking programming has been selected to support concurrency. The reason is because the python orchestrator is not CPU intensice and spents most of the time waiting the DB to response. Multiple processes could not work since not all functionalities are concurrent safe in MonetDB.
Threads are inappropriate because 1) we are not CPU-intensive so that we need more CPUs 2) Due to Python's GIL only one thread is allowed to hold the control of the Python interpreter at a time 3) Due to limited concurrency support of some functionalities in MonetDB many locks are required if using threads.
3) To support concurrent async programming, Monetdb's Python client has been modified using Python's asynchronous sockets and some implementations from 
MySQL's python async client (https://github.com/aio-libs/aiomysql)
4) https://docs.google.com/document/d/1rgYoajy3LqJ5ogK8Dejkix-g6lqPwEZdLGOHvCidr9Q/edit in page 3 of this document the most major issues that need updates are described
5) The library is as light as possible to support easy deployment. Only extremely necessary dependencies need to be installed (tornado, monetdb, numpy). The project is written mainly using the standard library.

<br>

<b>Todo:</b> <br>

1) Fault tolerance - check what happens when a node is down. Probably, a query using a connection object will raise a connection error exception, and then this node should be removed from the pool (connections.py file).
2) Online adding/removing nodes is supported but in the current version this is done by reinitializing everything. In the commented code in connections.py file
there is some code that tries to edit just the updates (and not re-init all the nodes) but there is a bug somewhere so this is commented. The bug happened when an update involves changing from monetdb to postgres or vice versa.  This should consider also concurrent requests in case of node updates, since there may be a living connection from another concurrent request during the update. This issue is not addressed in the current version.
3) Global node failure -> assign another global
4) clean the databases when the server is abnormaly shut down (e.g., ctrl-c). Currently, the tables are dropped only when the request returns some result or some error.
5) Set the way that the developer defines the flow of the tasks. The algorithm developer should not have direct access to the internal methods and objects of the system. - DONE with python's generators
6) Security, DB passwords etc.
7) There is a minimal error handling but probably this will need some updates.
8) Support for more kinds of tasks. Currently local (runs a function to all the local nodes and merge their results) and global (run a function on the merged local results and send the result back to locals) is supported. More kinds of tasks need to be implemented in order to support all kinds of dataflows, some examples:
- localdirect: run a function to one local node and send the result K other nodes (where K = 1...N)
- partitionbroadcast: run a function to global node split the result to partitions and send the partitions to the local nodes. Useful for map/reduce tasks (not for MIP but for more generic use)
- replicate: copy a table from one node to another node (also not for MIP but for more generic use)

9) Solve monetdb issues mentioned here at page 3 https://docs.google.com/document/d/1rgYoajy3LqJ5ogK8Dejkix-g6lqPwEZdLGOHvCidr9Q/edit. The most important issues are the following 2:
- Monetdb remote tables reconnect to the remote database each time the remote table is used in one or more queries and this is very time-consuming. Initializing the connections a-priori like in Postgres fdw solves this issue and also enhances concurrency (connecting to the DB is a blocking task)
- Create tables concurrently. Currently, this does not work and we have either to keep 2 connection objects per db (one concurrent and one blocking) or adding expensive locks. Monetdb should not just run them sequentially because in case we have a query like "create table glob as select pythonudf(col1) from table", we don't want the full query to run sequentially. We want the select part to run concurrently and the create part to run sequentially inside the db. This way the algorithm developer is able to avoid to define and return a static schema for each execution step and each step is able to return dynamic schemata. Concurrent create table also would lead to much simpler code, since there will be no need in waiting to get the schema in order to define/initialize the remote/merge tables creation and given that the code does not care about the returned schema this will lead to smaller and more readable code.

10) System speific tasks. Deamons, connectivity whatever...
11) Authentication? unique session/user ids?
12) Code clean-up (e.g., revisit the way that dbms specific functions are implemented - currently in the connection object of global node)
13) Dockers


<b>Research issues (probably not part of mvp):</b><br>
Monetdb:<br>
1) What happens if data is bigger than memory (e.g., chunking)
2) Balance between SQL and python
3) Languge issues (e.g., dynamic schema, yesql, how a udf is defined, how is it called and integrated in a python algorithm). Make the whole staff more user friendly.
4) Avoid copies of data in python udfs
5) Support of PyPy UDFs (in some aggregations with group by, pypy seems to be the only solution to avoid data copies and still run fast while writing python)



Orchestration:<br>
1) Dataflow language. Define a language to produce easy dataflows.
2) An abstraction to implement an understandable federated algorithm, both local/global calculations and dataflow in one script.
3) Support of more generic functionalities for distributed OLAP



