## Procedure

1. Run MonetDB with docker. <br/>
```
docker run -d -P -p 50000:50000 --name monetdb-1 thanasulas/monetdb:11.37.11
```
2. Run RabbitMQ with docker. <br/>
```
sudo docker run -d -p 5672:5672 --name rabbitmq-1 rabbitmq
```

3. Configure RabbitMQ. <br/>
Wait until RabbitMQ container is up and then start the configuration
```
sudo docker exec -it rabbitmq-1 rabbitmqctl add_user user password &&
sudo docker exec -it rabbitmq-1 rabbitmqctl add_vhost user_vhost &&
sudo docker exec -it rabbitmq-1 rabbitmqctl set_user_tags user user_tag &&
sudo docker exec -it rabbitmq-1 rabbitmqctl set_permissions -p user_vhost user ".*" ".*" ".*"
```

4. Inside the MIP-Engine folder run the celery worker: <br/>
```
celery -A worker.worker worker --loglevel=info
```

5. Run the tests <br/>
```
python3 ./worker/tests/tables.py
```