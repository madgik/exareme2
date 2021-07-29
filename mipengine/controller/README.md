## CONTROLLER service

### Build docker image

To build a new image you must be on the project root `MIP-Engine`, then

```
docker build -t <USERNAME>/mipengine_controller:<IMAGETAG> -f mipengine/controller/Dockerfile .
```

## Run

Then run the following command with your own env variables:

```
docker run -d --name controller -p 5000:5000 -e NODE_REGISTRY_IP=172.17.0.1 -e NODE_REGISTRY_PORT=8500 hbpmip/mipengine_controller:0.1
```
