## NODE service

### Build docker image

To build a new image you must be on the project root `MIP-Engine`, then

```
docker build -t <USERNAME>/mipengine_node:<IMAGETAG> -f mipengine/node/Dockerfile .
```

## Run

Then run the following command with your own env variables:

```
docker run -d --name globalnode -e NODE_IDENTIFIER=globalnode -e NODE_ROLE=GLOBALNODE -e LOG_LEVEL=INFO -e NODE_REGISTRY_IP=172.17.0.1 -e NODE_REGISTRY_PORT=8500 -e RABBITMQ_IP=172.17.0.1 -e RABBITMQ_PORT=5670 -e MONETDB_IP=172.17.0.1 -e MONETDB_PORT=50000 hbpmip/mipengine_node:0.1
```

## Run with env file

Create an env_file with the following variables:

```
NODE_IDENTIFIER=globalnode
NODE_ROLE=GLOBALNODE
LOG_LEVEL=INFO
NODE_REGISTRY_IP=172.17.0.1
NODE_REGISTRY_PORT=8500
RABBITMQ_IP=172.17.0.1
RABBITMQ_PORT=5670
MONETDB_IP=172.17.0.1
MONETDB_PORT=50000
```

Then start the container with:

```
docker run -d --name <CONTAINER_NAME> --env-file=.env_file hbpmip/mipengine_node:0.1
```
