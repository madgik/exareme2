The code is a kind of python/pseudocode(obviously not executable..). The reason I uploaded it is just to describe the approach I have in mind. 

A python wrapper layer that takes care of the underlying nodes databases manipulation, giving the algorithms* developer a simple, minimal interface to control the accessibility and flow of the data between her code and the several databases. Introducing such a layer is towards the "separation of concerns" principle which will lead in a more modular system, on which one can implement more complete testing.

In more detail, inside all the databases involved there will be a set of SQL user defined functions (python/sql/whatever..) implementing mathematical manipulations that will constitute the building blocks for more complex algorithms. The udfs must be minimal and general enough to encourage reuse in different algorithms.

In my example I have defined 4 wrapper functions that are calling underlying Modetdb udfs:

<pre><code>
def generate_random(params,node_native,nodes_broadcast):
def means_by_index( indices, datapoints, data_node, nodes_broadcast)
def min_column(table_name,node_native,nodes_broadcast):
def calculate_norm(points_a_table_name,points_b_table_name,node_native,nodes_broadcast):
</code></pre>

the last 2 parameters in all these functions define 
    1.the database in which the resulting table will be created and 
    2.the database to which the resulting table will be visible to(in case of monetdb via remote tables)

These functions in the wrapper have 3 purposes:
1.Calling the underlying SQL udfs
2.Dealing with the creation of the resulting table in the correct database, as defined by the caller of the function and
3.Dealing with the creation of remote tables that will make the resulting table visible to other databases as defined by the caller of the function

These function will return the name of the table that contains the results of the udfs. Since the caller defines which dbs will have access to the table, the table table name is enough to proceed with the data flow of the algorithm. That obviously means that the wrapper functions ought to guarantee that the returned table is accessible to the nodes(native and broadcast nodes) that the caller of the wrapper function defined

The idea is that the algorithm developper will only have to state the udf she wants to call, the name of resulting table, the db where the resulting table will be created and which other dbs should be able to see the resulting table

One implication of such schema is obviously that the developper writting an algorithm, in Python has to think in terms of database tables that exist only inside the dbs of the system (instead of readily availiable data in memory) and a predefined set of SQL udfs. Results returned from the udf wrapping functions are only table names, not the actual data (but can easily be exported to numpy array? a udf for that for debugging purposes??)
  
*From now on wherever I use the term algorithm I mean a Statistical/Machine Learning/Data Mining algorithm.


Not addressed in the current code:
    1.drop unused tables mechanism
    2.parallelism  This code is not using any asynchronicity at all, everything is executed sequentially. Nevertheless, a mechanism that will allow executing coroutines asynchronously is plausible.
    3.error handling

Pros:
    1.small general udfs, as they accumulate (wisely..) at some point (hopefully) no new ones will be needed for new algorithms implementations
    2.small general udfs, easy to test
    3.easier mocking of the calls to the actual db for testing the algorithm
    4.udfs can be initially written in Python, but since they ought to be small and general, someone else with better sql skills than the algorithm developper, can independently write them in native SQL for efficiency
    5...all other efficiency benefits from manipulating data in the db instead of in memory

Cons:
    1.When writting an algorithm you have to think in terms of database tables and predefined set of udfs, instead of readily availiable data in memory. Results returned from the udfs wrappers are only table names, not the actual data (but can easily be exported to numpy array? a udf for that for debugging purposes??)
    2.udfs must be VERY WELL DOCUMENTED and maintained, otherwise the set of udfs will become bloated and with overlapping functionality. Cleaning this later will be a nightmare since each algorithm design will have specific dependencies on them..
    3...and probably some more that I haven't thought about yet
