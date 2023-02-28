## NODE service

### Build docker image

To build a new image you must be on the project root `MIP-Engine`, then

```
docker build -t <USERNAME>/mipengine_node:<IMAGETAG> -f mipengine/node/Dockerfile .
```

## Run with env file

Create an env_file with the following variables:

```
NODE_IDENTIFIER=globalnode
NODE_ROLE=GLOBALNODE
LOG_LEVEL=INFO
FRAMEWORK_LOG_LEVEL=INFO
RABBITMQ_IP=172.17.0.1
RABBITMQ_PORT=5670
MONETDB_IP=172.17.0.1
MONETDB_PORT=50000
MONETDB_PASSWORD=executor
```

Then start the container with:

```
docker run -d --name <CONTAINER_NAME> --env-file=.env_file <USERNAME>/mipengine_node:<IMAGETAG>
```
