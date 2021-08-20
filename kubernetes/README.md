# MIP-Engine deployment with Kubernetes

## Configuration

The following packages need to be installed on **master/worker** nodes:
```
docker
kubelet
kubeadm
```

Packages needed on the **master** node only:
```
helm
```

To configure kubernetes to use docker you should also follow this [guide](https://kubernetes.io/docs/setup/production-environment/container-runtimes/#docker "guide") .


## Cluster Management

### Initialize the cluster

On the **master** node:
1) Run the following command to initialize the cluster:
```
sudo kubeadm init --pod-network-cidr=192.168.0.0/16
```

2) To enable kubectl run the following commands as prompted from the previous command:
```
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
```

3) Add calico network tool in the cluster:
```
kubectl apply -f https://docs.projectcalico.org/v3.20/manifests/calico.yaml
```

4) Allow master-specific pods to run on the **master** node with:
```
kubectl taint nodes --all node-role.kubernetes.io/master-
kubectl label node <master-node-name> nodeType=master
```

### Add a worker node to the cluster

1) On the **master** node, get the join token with the following command:
```
kubeadm token create --print-join-command
```

Use the provided on the **worker** node, with `sudo`, to join the cluster.

2) Allow worker-specific pods to run on the **worker** node with:
```
kubectl label node <worker-node-name> nodeType=worker
```


### Remove a worker node from the cluster

On the **master** node execute the following commands:
```
kubectl drain <node-name> --ignore-daemonsets
kubectl delete node <node-name>
```


## Deploy MIP-Engine

1) Configure the [helm chart values](values.yaml).
	- The `mipengine_images -> version` should be the mip-engine services version in dockerhub.
	- The `cdes_path` should be set to the master node hostpath that contains the pathologies metadata.
	- The `localnodes` is a counter for the localnodes. Should be equal to the number of local nodes that exist in the cluster.

2) From the `MIP-Engine` folder, deploy the services:
```
helm install mipengine kubernetes
```

### Change the MIP-Engine version running

1) Modify the `mipengine_images -> version` value in the [helm chart values](values.yaml) accordingly.

2) Upgrade the helm chart with:
```
helm upgrade mipengine kubernetes
```

### Increase/reduce the number of local nodes

1) Modify the `localnodes` value in the [helm chart values](values.yaml) accordingly.

2) Upgrade the helm chart with:
```
helm upgrade mipengine kubernetes
```

### Restart the federation
You can restart the federation with helm by running:
```
helm uninstall mipengine
helm install mipengine kubernetes
