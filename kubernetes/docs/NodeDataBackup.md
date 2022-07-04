## Backup of Node Data

The global/local Node data are stored in the `monetdb_storage` path defined in the [helm chart values](../values.yaml). This is the same for all the nodes.

In order to backup the global/local node data,

1. Get the pod id of the node to backup:

```
kubectl get pods -o wide
```

2. Stop the pod's database:

```
kubectl exec -i <pod_id> --container mipengine-monetdb -- monetdb lock db
kubectl exec -i <pod_id> --container mipengine-monetdb -- monetdb stop db
```

3. You can then copy the contents in the `monetdb_storage` of the monetdb and store them.

1. Start the pod's database again:

```
kubectl exec -i <pod_id> --container mipengine-monetdb -- monetdb start db
```
