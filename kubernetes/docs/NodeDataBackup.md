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

The global and local Worker data are now stored in the paths defined in the [helm chart values](../values.yaml). For the global node, data is stored under `db.globalworker_location`, and for the local node, it is stored under `db.localworker_location`.

#### In order to restore the instance of the database:

1. Move the database snapshot to your local node’s storage path defined by `db.localworker_location` in the [helm chart values](../values.yaml). Place the snapshot in the `db` subdirectory (for example, `<db.localworker_location>/db`).

1. Get the pod ID of the node to restore:

```
kubectl get pods -o wide
```

3. Restore the database's snapshot:

```
kubectl exec -i <pod_id> -c monetdb /bin/sh -c 'mclient db  <$MONETDB_STORAGE/db.sql'
```
