With Monetdb and python3 already installed

<ol>
<li> Navigate to <code>./sqlScripts</code> and execute createDBs.sh which will create <code>globaldb,localdb1,localdb2</code> databases and will also create a small "data" table in the 2 local databases (call the script from inside <code>./sqlScript</code> folder). <b>Pass</b>: monetdb
<pre><code>$sh ./createDBs.sh</code>
</li>

<li> In <code>./servers.py</code> set the url and port of your monetdb servers
</li>

<li> Start the server:
<pre><code>$python3 ./mserver.py</code></pre>
</li>

<li>make an http request
<pre><code>$python3 ./postRequest_scr_dummy.py</code></pre>
</li>
</ol>
<br>

<ol>
<li><h4><code> ..._udfs.py</code></h4> 
The part of the algorithm code that needs to be executed inside Monet, as python UDFs, will go here
</li>
<li><h4><code> ..._flow.py</code></h4>  
The rest of the code of the algorithm goes here. This part "orchestrates" how, when and where(on which DBs the UDFs will be called)
</li>
</ol>
<br>

<h2>Examples</h2>
<ul>
    <li><h3> Dummy algorithm</h3>
    
    <h4><code>dummy_udfs.py</code></h4>
    Two "dummy" UDFs implemented for demonstration purposes. Both UDFs (in <code>dummy_udfs.py</code>) just return their input. 
    <h4><code>dummy_flow.py</code></h4>
    <ol>
        <li>executes <code>local_calc</code> on all local nodes and stores results to tables on each local node. The input to the UDF is the table passed in the <code>postRequest_dummy_script.py</code> JSON object</li>
        <li> makes the tables "visible" to the global node, as a remote table and merges them to one table</li>
        <li>executes <code>global_calc</code> on the global nodes and stores result to a tables in the global node. The input to the UDF is the merged table of the previous step</li>
        <li>makes the table "visible" to all the local nodes, as a remote table</li>
    </ol>
    <h4><code>postRequest_dummy_script.py</code></h4>
    A python script to make the post request to the mserver to execute the dummy algorithm. A JSON object contains the names of the relevant files (<code>dummy_flow.py, dummy_udfs.py</code>) along with attributes needed for the execution of the algorithm, here just the name of the table containing the data the algorithm will run on.
    </li>
    <li><h3>Kmeans algorithm</h3>
    <strong><u>(This is not yet fully implemented and functional, here is an outline of the flow)</u></strong>
    <h4><code>kmeans_udfs.py</code></h4>
    Three python functions are defined here:
    <ul>
        <li><code>generate_initial_centroids</code>
            <p>Generates an initial set of centroids </p> 
        </li>
        <li><code>local_calc</code>
            <p>Assigns datapoints to a given set of centroids</p>
        </li>
        <li><code>global_calc</code>
            <p>Takes the assigned datapoints info and produces new centroids</p>
        </li>
    </ul>
    <h4><code>kmeans_flow.py</code></h4>
    <ol>
        <li>executes <code>generate_initial_centroids</code> on the global node, whihc genarates the coordinates of the initial centroids and stores the result to table on the global node. The input to the UDF is the table passed in the <code>postRequest_kmeans_script.py</code> JSON object</li>
        <li> makes the table "visible" to all the local nodes, as a remote table
        </li>
        <li>executes <code>local_calc</code> on all the local nodes and stores results to tables in the local nodes. The input to the UDF is the remote table of the previous step</li>
        <li>makes the local tables "visible" to the global node, as remote tables and merges them to one table</li>
        <li>executes <code>global_calc</code> on the global node and the new centroids coordinates are stored to a table in the global node.</li>
        <li>iterate from step 1.</li>
    </ol>
    <h4><code>postRequest_kmeans_script.py</code></h4>
    A python script to make the post request to the mserver to execute the kmeans algorithm. The JSON object contains the names of the relevant files (<code>kmeans_flow.py, kmeans_udfs.py</code>) along with attributes needed for the execution of the algorithm, here the name of the table containing the data the algorithm will run on, the number of clusters, the number of iterations and a min and max value for the randomly generated initial centroids.
</li>
</ul>
<br>
<h3>NOTES:</h3>
<ul>
<li>Every call to a UDF produces a table on the database where it is executed. The runtime takes care of uniquely naming the tables to avoid conflicts inside the database, from tables created from queries executed in the database, as well as tables created via the <i>remote table mechanism</i></li>
</ul>

