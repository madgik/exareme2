## Import data in the local nodes

### Setting up the data

Before the data manager can load data in a local node, ensure that the data is placed in the csvs subdirectory of the path specified by `db.localworker_location` in your [helm chart values](../values.yaml).
For example, if your `db.localworker_location` is set to `/opt/exareme2/localworker`, then the data should be located in `/opt/exareme2/localworker/csvs`.
That folder should contain the metadata of the data model that will be loaded and the dataset csvs.

### Importing the data in the node

The system administrator should first locate the localnode pod,
in the specific node that we want to import the data into.
The following command can be used, that shows the node where the pods are deployed:

```
kubectl get pods -o wide
```

The data manager using the `POD_ID`, can enter the pod in the `db-importer` container and use the `mipdb`
script to import data.

1. The db should be initialized first:

```
kubectl exec <POD_ID> -c db-importer -- sh -c 'mipdb init'
```

2. The data that have been setup previously will be located in the `/opt/data` folder inside the container.
   Before importing any csvs the data model should be added. For example:

```
kubectl exec <POD_ID> -c db-importer -- sh -c 'mipdb add-data-model /opt/data/dementia_v_0_1/CDEsMetadata.json'
```

3. You can then add datasets, in the data model that you added previously. For example:

```
kubectl exec <POD_ID> -c db-importer -- sh -c 'mipdb add-dataset /opt/data/dementia_v_0_1/edsd0.csv -d dementia -v 0.1'
```

4. There are many other commands that you can use. Check them out using the help:

```
kubectl exec <POD_ID> -c db-importer -- sh -c 'mipdb --help'
```

### (FAST) Import of all the data available

Instead of manually importing each data model and dataset there is a command that loads all of the data models and datasets inside a specific folder.
The to-be imported folder should contain one subfolder for each data model and inside of that any number of dataset csv files.

```
kubectl exec <POD_ID> -c db-importer -- sh -c 'mipdb init'
kubectl exec <POD_ID> -c db-importer -- sh -c 'mipdb load-folder /opt/data'
```

**ATTENTION** Only absolute paths can be used with this command.
