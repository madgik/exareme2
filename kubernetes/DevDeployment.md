# Exareme2 Development deployment with Kubernetes in one node

## Configuration

The following packages need to be installed:

```
docker
kubectl
helm
```

kubectl [installation guide (Ubuntu)](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/)<br />
helm [installation gude](https://helm.sh/docs/intro/install/)

## Data preparation (Temporary Solution)

MonetDB needs to have the data loaded from a volume and not imported due to a memory leak. So, the idea is you load the data to a temporary container running monetdb, then copy the monetdb's produced files to the folders that will be mounted as volumes for the "real" monetdb containers for each worker of the system.

1. Start a monetdb container:

```
sudo rm -rf /opt/monetdb
docker run --name monetdb_tmp -d -p 50010:50000 -v /opt/monetdb:/home/monetdb madgik/exareme2_db:latest
```

2. Load the data into that container:
   <br />inside the root folder of the project `Exareme2/`

```
poetry run inv load-data --port 50010
```

3. Lock and stop the db so the data can be exported:

```
docker exec -i monetdb_tmp monetdb lock db
docker exec -i monetdb_tmp monetdb stop db
```

4. Copy the data produced to the volumes that will be used from kuberentes monetdb's:

```
sudo rm -rf /opt/monetdb1 /opt/monetdb2
sudo cp -r /opt/monetdb/. /opt/monetdb1/
sudo cp -r /opt/monetdb/. /opt/monetdb2/
```

5. Remove the used monetdb container:

```
docker stop monetdb_tmp && docker rm monetdb_tmp
```

## Setup the kubernetes cluster with kind

1. Delete the existing cluster, if it exists

```
kind delete cluster
```

2. Create the cluster using the prod_env_tests setup (you can create a custom one if you want) :

```
kind create cluster --config tests/prod_env_tests/kind_configuration/kind_cluster.yaml
```

3. After the nodes are started, you need to taint them properly:
   <br /><br />The following 2 sets of commands taint and label the master (kubernetes) node. Both command sets, taint and label the same node (master kubernetes node), referenced with different name due to `kind` naming the nodes differently from time to time.

<br />Run command set 1:

```
kubectl taint nodes kind-control-plane node-role.kubernetes.io/master-
kubectl label node kind-control-plane nodeType=master
```

If the previous `kubectl` command gives an error about `kind-control-plane` not found, then run the command set 2:

```
kubectl taint nodes master node-role.kubernetes.io/master-
kubectl label node master nodeType=master
```

(you can also get the existing nodes with `kubectl get nodes`)<br />
<br />Taint and label the worker nodes

```
kubectl label node worker1 nodeType=worker
kubectl label node worker2 nodeType=worker
```

4. (Optional) Build and load the docker images for the kuberentes cluster. If this step is ommited, the images will be pulled from dockerhub, which will most likely be slower, depending on the speed of your connection

First, build the images:
<br />(you can execute these in separate terminals, concurrently)

```
docker build -f monetdb/Dockerfile -t madgik/exareme2_db:latest ./
docker build -f rabbitmq/Dockerfile -t madgik/exareme2_rabbitmq:latest ./
docker build -f exareme2/worker/Dockerfile -t madgik/exareme2_worker:latest ./
docker build -f exareme2/controller/Dockerfile -t madgik/exareme2_controller:latest ./
```

Second, load the docker images to the kuberentes cluster

```
kind load docker-image madgik/exareme2_db:latest
kind load docker-image madgik/exareme2_rabbitmq:latest
kind load docker-image madgik/exareme2_worker:latest
kind load docker-image madgik/exareme2_controller:latest --nodes kind-control-plane
```

5. Deploy the Exareme2 kubernetes pods using helm charts:

```
helm install exareme2 kubernetes/
```

6. (Validation) You can then run the prod_env_tests to see if the deploymnt is working (it might take about a minute for services to sync):

```
poetry run pytest tests/prod_env_tests/
```
