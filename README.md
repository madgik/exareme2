# monetdb_federated_poc

<b>Installation</b>
1) Python3 with numpy should be installed in all federation nodes. The mserver runs with Python 3.7 or newer. It does not run with python3 versions older than 3.7.
2) Install monetdb from source (https://www.monetdb.org/Developers/SourceCompile) to all the nodes of the federation
   dependencies: `sudo apt-get install libssl-dev` `sudo apt-get install libpcre3 libpcre3-dev` `sudo apt-get install pkg-config` `sudo apt-get install python3-dev` `sudo apt-get install uuid-dev` `sudo apt-get install libxml2 libxml2-dev`
3) Install dependencies for mserver: `pip3 install tornado` , `pip3 install pymonetdb`
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
6) Run udfs.sql file in `mclient` in all the monetdb databases.
7) Include in servers.py file all the global/local nodes (as in the already existing example). The first node is the global.
8) Default port for mserver is hard-coded in mserver.py file.

<b>Usage:</b> 
Run server: <br>

`python3 mserver.py`


<b>URL Request Post:</b> <br>
<br> Content-Type: application/x-www-form-urlencoded (usually the default in most libraries), type application/json is not supported in tornado web framework</br>
Two fields: <br>
<br> 1) `algorithm` (e.g., "pearson")
<br> 2) `params`: valid json including table name, attributes and filters. e.g. Filters follow the DNF (disjunctive normal form:
The innermost tuples each describe a single column predicate. The list of inner predicates is interpreted as a conjunction (AND), forming a more selective and multiple column predicate. Finally, the most outer list combines these filters as a disjunction (OR).
`{"table":"data", "attributes":["c1","c2"],"parameters":[0.7,4],"filters":[[["c1",">","2"],["c1","<","10000"]],[["c1",">","0"]]]}`
(i.e., pearson requires a table with 2 float attributes)

<br>
<b>Implement a new algorithm:</b> <br>

1) Add its UDFs to udf.sql file
2) Add its lib to algorithms folder
3) Add an [algorithm name].py file to algorithms folder which returns the sql query for each step of the algorithm
4) Update schema.json file accordingly

<br>
<b>Other features:</b> <br>

1) Updating servers.py file the module is auto reloaded online and does not require restarting
2) A simple fault tolerance has been added for local nodes.

<br>

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

1) Evaluate fault tolerance and make it more robust
2) Global node failure -> assign another global
3) Security, monetdb passwords etc.

<b>Research issues (probably not part of mvp):</b><br>
Monetdb:<br>
1) What happens if data is bigger than memory (e.g., chunking)
2) Balance between SQL and python
3) Avoid copies of data in python udfs
4) Support of PyPy UDFs (in some aggregations with group by, pypy seems to be the only solution to avoid data copies and still run fast while writing python)
5) Languge issues (e.g., dynamic schema, yesql)


Federation:<br>
1) Dataflow language. Define a language to produce easy dataflows.
2) An abstraction to implement an understandable federated algorithm, both local/global calculations and dataflow in one script.



