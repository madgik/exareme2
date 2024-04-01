## CONTROLLER service

### Build docker image

To build a new image you must be on the project root `Exareme2`, then

```
docker build -t <USERNAME>/exareme2_controller:<IMAGETAG> -f exareme2/controller/Dockerfile .
```

## Run with env file

Create a file with the workers' locations, for example:

```
["172.17.0.1:5670", "172.17.0.1:5671", "172.17.0.1:5672"]
```

Create an env_file with the following variables:

```
LOG_LEVEL=INFO
FRAMEWORK_LOG_LEVEL=INFO
DEPLOYMENT_TYPE=LOCAL
WORKER_LANDSCAPE_AGGREGATOR_UPDATE_INTERVAL=30
LOCALWORKERS_CONFIG_FILE=/home/user/localworkers_config.json
```

Then start the container with:

```
docker run -d --name <CONTAINER_NAME> --env-file=.env_file <USERNAME>/exareme2_controller:<IMAGETAG>
```
