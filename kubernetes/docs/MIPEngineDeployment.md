## Deploy MIP-Engine

1. Configure the [helm chart values](../values.yaml).

   - The `mipengine_images -> version` should be the mip-engine services version in dockerhub.
   - The `localnodes` is a counter for the localnodes. Should be equal to the number of local nodes that exist in the cluster.

1. From the `MIP-Engine` folder, deploy the services:

```
helm install mipengine kubernetes
```

**For a deployment with microk8s use `microk8s helm3` in the commands.**

### Change the MIP-Engine version running

1. Modify the `mipengine_images -> version` value in the [helm chart values](../values.yaml) accordingly.

1. Upgrade the helm chart with:

```
helm upgrade mipengine kubernetes
```

### Increase/reduce the number of local nodes

1. Modify the `localnodes` value in the [helm chart values](../values.yaml) accordingly.

1. Upgrade the helm chart with:

```
helm upgrade mipengine kubernetes
```

### Restart the federation

You can restart the federation with helm by running:

```
helm uninstall mipengine
helm install mipengine kubernetes
```

## Log Rotation Configuration

In order to avoid docker logs taking up too much space in the **master**/**worker** nodes, the docker engine should be configured appropriately in every node.

[Docker-Engine Log Rotation Guide](https://docs.docker.com/config/containers/logging/configure/#configure-the-default-logging-driver)
