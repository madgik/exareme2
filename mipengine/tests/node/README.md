## Procedure

1. Run MonetDB with docker. <br/>
```
docker run -d -P -p 50000:50000 --name monetdb-1 thanasulas/monetdb:11.37.11
docker run -d -P -p 50001:50000 --name monetdb-2 thanasulas/monetdb:11.37.11
```
2. Run RabbitMQ with docker. <br/>
```
sudo docker run -d -p 5672:5672 --name rabbitmq-1 rabbitmq
sudo docker run -d -p 5673:5672 --name rabbitmq-2 rabbitmq
```

3. Configure RabbitMQ. <br/>
Wait until RabbitMQ container is up and then start the configuration
```
sudo docker exec -it rabbitmq-1 rabbitmqctl add_user user password &&
sudo docker exec -it rabbitmq-1 rabbitmqctl add_vhost user_vhost &&
sudo docker exec -it rabbitmq-1 rabbitmqctl set_user_tags user user_tag &&
sudo docker exec -it rabbitmq-1 rabbitmqctl set_permissions -p user_vhost user ".*" ".*" ".*"
Same for rabbitmq-2
```

4. Inside the MIP-Engine folder run the celery workers: <br/>
```
python3 mipengine/tests/node/worker_config_setup.py 50000 5672 && celery -A mipengine.node.node worker --loglevel=info
python3 mipengine/tests/node/worker_config_setup.py 50001 5673 && celery -A mipengine.node.node worker --loglevel=info
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