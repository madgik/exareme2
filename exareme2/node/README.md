## NODE service

### Build docker image

To build a new image you must be on the project root `Exareme2`, then

```
docker build -t <USERNAME>/exareme2_node:<IMAGETAG> -f exareme2/node/Dockerfile .
```

## Run with env file

Create an env_file with the following variables:

```
NODE_IDENTIFIER=globalnode
NODE_ROLE=GLOBALNODE
LOG_LEVEL=INFO
FRAMEWORK_LOG_LEVEL=INFO
PROTECT_LOCAL_DATA=false
RABBITMQ_IP=172.17.0.1
RABBITMQ_PORT=5670
MONETDB_IP=172.17.0.1
MONETDB_PORT=50000
MONETDB_LOCAL_USERNAME=executor
MONETDB_LOCAL_PASSWORD=executor
MONETDB_PUBLIC_USERNAME=guest
MONETDB_PUBLIC_PASSWORD=guest
SMPC_ENABLED=false
```

Then start the container with:

```
docker run -d --name <CONTAINER_NAME> --env-file=.env_file <USERNAME>/exareme2_node:<IMAGETAG>
```
