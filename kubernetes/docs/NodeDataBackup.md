## Backup of Node Data

#### In order to backup the global/local node data:

1. Get the pod id of the node to backup:

```
kubectl get pods -o wide
```

2. Create a snapshot of the database:

```
kubectl exec -i <pod_id> -c monetdb -- sh -c 'msqldump --database=db > $MONETDB_STORAGE/db.sql'
```

The global/local Node data are stored in the `db.storage_location` path defined in the [helm chart values](../values.yaml). This is the same for all the nodes.

#### In order to restore the instance of the database:

1. Move the database snapshot to your local storage `db.storage_location` path defined in the [helm chart values](../values.yaml).

1. Get the pod id of the node to restore:

```
kubectl get pods -o wide
```

3. Restore the database's snapshot:

```
kubectl exec -i <pod_id> -c monetdb /bin/sh -c 'mclient db  <$MONETDB_STORAGE/db.sql'
```
