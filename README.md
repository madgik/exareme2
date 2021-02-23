# MIP-Engine

## Installation

1. Install python <br/>
```
sudo apt install python3.8
sudo apt install python3-pip
```

### Controller API

1. Install requirements. <br/>
```
python3.8 -m pip install -r ./requirements/controller.txt 
```

2. Run the controller API. <br/>
```
export QUART_APP=mipengine/controller/api/app:app; python3.8 -m quart run
```

### Nodes setup

1. Deploy MonetDB (docker). <br/>
```
docker run -d -P -p 50000:50000 --name monetdb-1 thanasulas/monetdb:11.37.11
docker run -d -P -p 50001:50000 --name monetdb-2 thanasulas/monetdb:11.37.11
```

2. Import the csvs in MonetDB Inside the MIP-Engine folder, to import all the csvs on both dbs, run:
```
python3 mipengine/node/monetdb_interface/csv_importer.py -folder /home/thanasis/Desktop/MIP-Engine/mipengine/tests/data/ -user monetdb -pass monetdb -url localhost:50000 -farm db
python3 mipengine/node/monetdb_interface/csv_importer.py -folder /home/thanasis/Desktop/MIP-Engine/mipengine/tests/data/ -user monetdb -pass monetdb -url localhost:50001 -farm db
```

3. Deploy RabbitMQ (docker). <br/>
```
sudo docker run -d -p 5672:5672 --name rabbitmq-1 rabbitmq
sudo docker run -d -p 5673:5672 --name rabbitmq-2 rabbitmq
```

4. Configure RabbitMQ. <br/>
   Wait until RabbitMQ containers are up and then run the configuration.
```
sudo docker exec -it rabbitmq-1 rabbitmqctl add_user user password &&
sudo docker exec -it rabbitmq-1 rabbitmqctl add_vhost user_vhost &&
sudo docker exec -it rabbitmq-1 rabbitmqctl set_user_tags user user_tag &&
sudo docker exec -it rabbitmq-1 rabbitmqctl set_permissions -p user_vhost user ".*" ".*" ".*" &&
sudo docker exec -it rabbitmq-2 rabbitmqctl add_user user password &&
sudo docker exec -it rabbitmq-2 rabbitmqctl add_vhost user_vhost &&
sudo docker exec -it rabbitmq-2 rabbitmqctl set_user_tags user user_tag &&
sudo docker exec -it rabbitmq-2 rabbitmqctl set_permissions -p user_vhost user ".*" ".*" ".*"
```

5. Install requirements. <br/>
```
python3.8 -m pip install -r ./requirements/node.txt
```

6. Inside the MIP-Engine folder setup the data tables: <br/>
```
python3 mipengine/tests/node/data/setup_data_tables.py
```

7. Inside the MIP-Engine folder run the celery workers: <br/>
```
python3 mipengine/tests/node/set_node_identifier.py local_node_1 && celery -A mipengine.node.node worker --loglevel=info
python3 mipengine/tests/node/set_node_identifier.py local_node_2 && celery -A mipengine.node.node worker --loglevel=info
```

## Tests

1. Install requirements <br/>
```
sudo apt install python3.8
sudo apt install tox
```

2. Run the tests <br/>
```
tox
```