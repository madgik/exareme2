## WORKER service

### Build docker image

To build a new image you must be on the project root `Exareme2`, then

```
docker build -t <USERNAME>/exareme2_worker:<IMAGETAG> -f exaflow/worker/Dockerfile .
```

## Run with env file

Create an env_file with the following variables:

```
WORKER_IDENTIFIER=globalworker
WORKER_ROLE=GLOBALWORKER
LOG_LEVEL=INFO
FRAMEWORK_LOG_LEVEL=INFO
DUCKDB_PATH=/opt/data/globalworker.duckdb
DATA_PATH=/opt/data/
CONTROLLER_IP=172.17.0.1
CONTROLLER_PORT=5000
PROTECT_LOCAL_DATA=false
GRPC_IP=172.17.0.1
GRPC_PORT=5670
SMPC_ENABLED=false
```

If `DUCKDB_PATH` is omitted the worker stores its DuckDB file under
`$DATA_PATH/<WORKER_IDENTIFIER>.duckdb`.

Then start the container with:

```
docker run -d --name <CONTAINER_NAME> --env-file=.env_file <USERNAME>/exareme2_worker:<IMAGETAG>
```
