# MIP-Engine Development deployment with Kubernetes in one node

## Configuration

The following packages need to be installed:

```
docker
kubectl
helm
```

## (Optional) Image preparation

You can use locally built images to be deployed on kubernetes. First you should build the images:
```
docker build -f mipengine/monetdb/Dockerfile -t madgik/mipenginedb:latest
docker build -f mipengine/rabbitmq/Dockerfile -t madgik/mipengine_rabbitmq:latest
docker build -f mipengine/node/Dockerfile -t madgik/mipengine_node:latest .
docker build -f mipengine/controller/Dockerfile -t madgik/mipengine_controller:latest .
```

## Data preparation (Temporary Solution)

MonetDB needs to have the data loaded from a volume and not imported due to a memory leak.

1. Start a monetdb container:
```
sudo rm -rf /opt/monetdb
docker run --name monetdb -d -p 50000:50000 -v /opt/monetdb:/home/monetdb madgik/mipenginedb:latest
```

2. Load the data into that container:
```
poetry run inv load-data --port 50000
```

3. Lock and stop the db so the data can be exported:
```
docker exec -i monetdb monetdb lock db
docker exec -i monetdb monetdb stop db
```

4. Copy the data produced to the volumes that will be used from kuberentes monetdb's:
```
sudo rm -rf /opt/monetdb1 /opt/monetdb2
sudo cp -r /opt/monetdb/. /opt/monetdb1/
sudo cp -r /opt/monetdb/. /opt/monetdb2/ 
```

5. Remove the used monetdb container:
```
docker stop monetdb && docker rm monetdb
```

## Setup the kubernetes cluster with kind

1. Create the cluster using the e2e_tests setup (you can create a custom one if you want) :
```
kind create cluster --config tests/e2e_tests/kind_configuration/kind_cluster.yaml 
```

2. After the nodes are started, you need to taint them properly:
```
kubectl taint nodes kind-control-plane node-role.kubernetes.io/master-
kubectl label node kind-control-plane nodeType=master
kubectl taint nodes master node-role.kubernetes.io/master-
kubectl label node master nodeType=master
kubectl label node worker1 nodeType=worker
kubectl label node worker2 nodeType=worker
```
FYI: Some commands are duplicated due to kind naming the nodes differently from time to time.

3. (Optional) Load the docker images to the kuberentes cluster, if not the images will be pulled from dockerhub:
```
kind load docker-image madgik/mipenginedb:latest
kind load docker-image madgik/mipengine_rabbitmq:latest
kind load docker-image madgik/mipengine_node:latest
kind load docker-image madgik/mipengine_controller:latest --nodes kind-control-plane
```

4. Deploy the MIP-Engine kubernetes pods using helm charts:
```
helm install mipengine kubernetes/
```

5 (Validation) You can then run the e2e_tests to see if everything works properly:
```
poetry run pytest tests/e2e_tests/               
```
         
