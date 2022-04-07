## CONTROLLER service

### Build docker image

To build a new image you must be on the project root `MIP-Engine`, then

```
docker build -t <USERNAME>/mipengine_controller:<IMAGETAG> -f mipengine/controller/Dockerfile .
```

## Run with env file

Create a file with the nodes' locations, for example:

```
["172.17.0.1:5670", "172.17.0.1:5671", "172.17.0.1:5672"]
```

Create an env_file with the following variables:

```
LOG_LEVEL=INFO
FRAMEWORK_LOG_LEVEL=INFO
CDES_METADATA_PATH=172.17.0.1
DEPLOYMENT_TYPE=LOCAL
NODE_LANDSCAPE_AGGREGATOR_UPDATE_INTERVAL=30
LOCALNODES_CONFIG_FILE=/home/user/localnodes_config.json
```

Then start the container with:

```
docker run -d --name <CONTAINER_NAME> --env-file=.env_file <USERNAME>/mipengine_controller:<IMAGETAG>
```
