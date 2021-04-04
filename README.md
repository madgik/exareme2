# MIP-Engine

## Setup

### Prerequisites
Install [python3.8](https://www.python.org/downloads/ "python3.8")
```
sudo apt install python3.8
sudo apt install python3-pip
```

### Setup Environment
1. For everything that follows you need to be in the project's root directory, i.e. `MIP-Engine/`

2. Setup a [virtualenv](https://docs.python.org/3.8/tutorial/venv.html "virtual environment")

3. Set `PYTHONPATH`
   ```
   export PYTHONPATH=$(pwd):$PYTHONPATH
   ```
   Or, to avoid setting it in every new shell
   ```
   echo 'export PYTHONPATH=/path/to/MIP-Engine/:$PYTHONPATH' >> ~/.bashrc
   ```

4. Install all requirements
   ```
   python -m pip install -r requirements/algorithms.txt requirements/controller.txt requirements/node.txt requirements/tests.txt
   ```

5. The following script will set your local network IP in `mipengine/resources/node_catalog.json`.
   First you need to find your ip. On Linux/MacOS list all available ips with
   ```
   ifconfig | grep "inet "
   ```
   To get a particular interface's ip use
   ```
   ifconfig <INTERFACE> | awk '/inet / { print $2 }'
   ```
   Once you know your machine's ip set it in the node catalog with
   ```
   python mipengine/tests/node/set_hostname_in_node_catalog.py -host <IP>
   ```


### Nodes Setup
1. Kill existing monetdb and rabbitmq containers, if present
   ```
   docker ps -a | grep -E 'monet|rabbitmq' | awk '{ print $1 }' | xargs docker rm -vf
   ```

2. Start MonetDB containers
   ```
   docker run -d -P -p 50000:50000 --name monetdb-0 jassak/mipenginedb:dev1.1  # global node
   docker run -d -P -p 50001:50000 --name monetdb-1 jassak/mipenginedb:dev1.1  # local node 1
   docker run -d -P -p 50002:50000 --name monetdb-2 jassak/mipenginedb:dev1.1  # local node 2
   ```

3. Start RabbitMQ containers
   ```
   docker run -d -p 5670:5672 --name rabbitmq-0 rabbitmq  # global node
   docker run -d -p 5671:5672 --name rabbitmq-1 rabbitmq  # local node 1
   docker run -d -p 5672:5672 --name rabbitmq-2 rabbitmq  # local node 2
   ```

4. Populate the 2 local nodes databases from the csv data files
   ```
   python -m mipengine.node.monetdb_interface.csv_importer -folder ./mipengine/tests/data/ -user monetdb -pass monetdb -url localhost:50001 -farm db
   python -m mipengine.node.monetdb_interface.csv_importer -folder ./mipengine/tests/data/ -user monetdb -pass monetdb -url localhost:50002 -farm db
   ```

5. Configure RabbitMQ. *WARNING* RabbitMQ needs ~30 secs to be ready to execute the following commands.
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

6. Start nodes
   ```
   python mipengine/tests/node/set_node_identifier.py globalnode && celery -A mipengine.node.node worker --loglevel=info
   python mipengine/tests/node/set_node_identifier.py localnode1 && celery -A mipengine.node.node worker --loglevel=info
   python mipengine/tests/node/set_node_identifier.py localnode2 && celery -A mipengine.node.node worker --loglevel=info
   ```

### Controller Setup

1. Start controller
   ```
   export QUART_APP=mipengine/controller/api/app:app; python -m quart run
   ```

### Algorithm Run

1. Call the test script which performs a post request to the controller
   ```python test_post_request.py```
