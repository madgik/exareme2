With Monetdb and python3 already installed...<br><br>
1. Navigate to ./sqlScripts and execute createDBs.sh which will create globaldb,localdb1,localdb2 databases and will also create a small "data" table in the 2 local DBs (call the script from inside ./sqlScript). <b>Pass</b>: monetdb
<br><code>$sh ./createDBs.sh</code>
<br><br>
2. In ./servers.py set the url and port of your monetdb servers
<br><br>
3. Start the server:
<pre><code>$python3 ./mserver.py</code></pre>
<br>
4. make a http request
<pre><code>$python3 ./postRequest_scr_dummy.py</code></pre>

