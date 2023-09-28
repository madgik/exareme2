## Deploy Exareme2

1. Configure the [helm chart values](../values.yaml).

   - The `exareme2_images -> version` should be the exareme2 services version in dockerhub.
   - The `localnodes` is a counter for the localnodes. Should be equal to the number of local nodes that exist in the cluster.

1. (Optional) Configure each localnode with a custom db password, for increased inter-node security:
   - In each localnode go to the folder specified at `db -> credentials_location` in the helm chart values.
   - Create a file named: `monetdb_password.sh`
   - Append the following lines in that file after changing the password:
      ```
      #!/bin/bash
      export MONETDB_LOCAL_PASSWORD="worker1"
      ```
   - Make the file executable.
   
1. From the `Exareme2` folder, deploy the services:
```
helm install exareme2 kubernetes
```

**For a deployment with microk8s use `microk8s helm3` in the commands.**

### Change the Exareme2 version running

1. Modify the `exareme2_images -> version` value in the [helm chart values](../values.yaml) accordingly.

1. Upgrade the helm chart with:

```
helm upgrade exareme2 kubernetes
```

### Increase/reduce the number of local nodes

1. Modify the `localnodes` value in the [helm chart values](../values.yaml) accordingly.

1. Upgrade the helm chart with:

```
helm upgrade exareme2 kubernetes
```

### Restart the federation

You can restart the federation with helm by running:

```
helm uninstall exareme2
helm install exareme2 kubernetes
```

## Log Rotation Configuration

In order to avoid docker logs taking up too much space in the **master**/**worker** nodes, the docker engine should be configured appropriately in every node.

[Docker-Engine Log Rotation Guide](https://docs.docker.com/config/containers/logging/configure/#configure-the-default-logging-driver)
