## Procedure

1. Run MonetDB with docker. <br/>
```
docker run -d -P -p 50000:50000 --name monetdb-1 thanasulas/monetdb:11.37.11 &&
docker exec -it monetdb-1 mclient db
```
2. Run RabbitMQ with docker. <br/>
```
sudo docker run -d -p 5672:5672 --name rabbitmq-1 rabbitmq
```

3. Configure RabbitMQ with docker. <br/>
```
sudo docker exec -it rabbitmq-1 rabbitmqctl add_user user password &&
sudo docker exec -it rabbitmq-1 rabbitmqctl add_vhost user_vhost &&
sudo docker exec -it rabbitmq-1 rabbitmqctl set_user_tags user user_tag &&
sudo docker exec -it rabbitmq-1 rabbitmqctl set_permissions -p user_vhost user ".*" ".*" ".*"
```

4. Inside the MIP-Engine folder run the celery worker: <br/>
```
export CELERY_BROKER_PORT="5672" && celery -A worker.worker worker --loglevel=info
```

4. Run the tests <br/>
```
python3 ./worker/tests/tables.py
```