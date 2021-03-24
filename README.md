## Prerequisites

Install python <br/>
```
sudo apt install python3.8
sudo apt install python3-pip
```

## Setup
1. The following script will set your local network IP in the `mipengine/resources/node_catalog.json`:<br/>

```
#network module name is specific to the machine, so wlo1 can also be wlan0 or something else. Check your ifconfig..
ip4=$(/sbin/ip -o -4 addr list wlo1 | awk '{print $4}' | cut -d/ -f1) 
python3.8 mipengine/tests/node/set_hostname_in_node_catalog.py -host $ip4
```
or
```
python3.8 mipengine/tests/node/set_monetdb_hostname.py -host <local network IP>
```

### Start the nodes (3 nodes: 1 global node, 2 local nodes)
1. Kill existing monetdb and rabbitmq containers, if present <br/>
```
docker rm -f $(docker ps -q) #!!!WARNING: WILL KILL ALL EXISTING CONTAINERS
```

2. Navigate to `/MIP-Engine/` <br/>

2. Add the MIP-Engine folder to your PYTHONPATH<br/>
```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

4. Install node requirements. <br/>
```
python3.8 -m pip install -r ./requirements/node.txt
```

3. Start the dockerized MonetDB instances. <br/>
```
docker run -d -P -p 50000:50000 --name monetdb-0 jassak/mipenginedb:dev1.1 #global node
docker run -d -P -p 50001:50000 --name monetdb-1 jassak/mipenginedb:dev1.1 #local node 1
docker run -d -P -p 50002:50000 --name monetdb-2 jassak/mipenginedb:dev1.1 #local node 2
```

5. Populate the 2 local nodes databases from the csv data files.
```
python3.8 -m mipengine.node.monetdb_interface.csv_importer -folder ./mipengine/tests/data/ -user monetdb -pass monetdb -url localhost:50001 -farm db
python3.8 -m mipengine.node.monetdb_interface.csv_importer -folder ./mipengine/tests/data/ -user monetdb -pass monetdb -url localhost:50002 -farm db
```

2. Start the dockerized RabbitMQ instances. <br/>
```
docker run -d -p 5670:5672 --name rabbitmq-0 rabbitmq #global node
docker run -d -p 5671:5672 --name rabbitmq-1 rabbitmq #local node 1
docker run -d -p 5672:5672 --name rabbitmq-2 rabbitmq #local node 2
```

3. Configure RabbitMQ. <br/>
   !!!WARNING RabbitMQ needs ~30 secs to be ready to execute the following commands.
```
docker exec -it rabbitmq-0 rabbitmqctl add_user user password &&
docker exec -it rabbitmq-0 rabbitmqctl add_vhost user_vhost &&
docker exec -it rabbitmq-0 rabbitmqctl set_user_tags user user_tag &&
docker exec -it rabbitmq-0 rabbitmqctl set_permissions -p user_vhost user ".*" ".*" ".*" &&

docker exec -it rabbitmq-1 rabbitmqctl add_user user password &&
docker exec -it rabbitmq-1 rabbitmqctl add_vhost user_vhost &&
docker exec -it rabbitmq-1 rabbitmqctl set_user_tags user user_tag &&
docker exec -it rabbitmq-1 rabbitmqctl set_permissions -p user_vhost user ".*" ".*" ".*" &&

docker exec -it rabbitmq-2 rabbitmqctl add_user user password &&
docker exec -it rabbitmq-2 rabbitmqctl add_vhost user_vhost &&
docker exec -it rabbitmq-2 rabbitmqctl set_user_tags user user_tag &&
docker exec -it rabbitmq-2 rabbitmqctl set_permissions -p user_vhost user ".*" ".*" ".*" 
```

4. Start global node
```
python3.8 mipengine/tests/node/set_node_identifier.py globalnode && celery -A mipengine.node.node worker --loglevel=info
```
5. Start local node 1
```
(if in new shell) export PYTHONPATH=$PYTHONPATH:$(pwd)
python3.8 mipengine/tests/node/set_node_identifier.py localnode1 && celery -A mipengine.node.node worker --loglevel=info
```
6. Start local node 2
```
(if in new shell) export PYTHONPATH=$PYTHONPATH:$(pwd)
python3.8 mipengine/tests/node/set_node_identifier.py localnode2 && celery -A mipengine.node.node worker --loglevel=info
```

### Start the Controller

2. Navigate to `/MIP-Engine/` <br/>

2. Add the MIP-Engine folder to your PYTHONPATH<br/>
```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

2. Install controller requirements. <br/>
```
python3.8 -m pip install -r ./requirements/controller.txt
```

2. Run the controller. <br/>
```
export QUART_APP=mipengine/controller/api/app:app; python3.8 -m quart run
```

### Execute an algorithm

1. Navigate to `/MIP-Engine/` <br/>
2. Call the test script which performs a post request to the controller
```
python3.8 test_post_request.py
```
