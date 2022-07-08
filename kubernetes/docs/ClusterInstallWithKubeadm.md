## Cluster Installation with kubeadm

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

### Hostname configuration

Currently, each node is distinguished from another one using their hostnames and they <b>MUST</b> be alphanumeric.
<br>If the node's hostname does not comply with that convention it has to be configured
from the kubelet following this [guide](https://kubernetes.io/docs/reference/labels-annotations-taints/#kubernetesiohostname).

## Cluster Management

### Initialize the cluster

On the **master** node:

1. Run the following command to initialize the cluster:

```
sudo kubeadm init --pod-network-cidr=192.168.0.0/16
```

2. To enable kubectl run the following commands as prompted from the previous command:

```
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
```

3. Add calico network tool in the cluster:

```
kubectl apply -f https://docs.projectcalico.org/v3.20/manifests/calico.yaml
```

4. Allow master-specific pods to run on the **master** node with:

```
kubectl taint nodes --all node-role.kubernetes.io/master-
kubectl label node <master-node-name> nodeType=master
```

### Add a worker node to the cluster

1. On the **master** node, get the join token with the following command:

```
kubeadm token create --print-join-command
```

Use the provided on the **worker** node, with `sudo`, to join the cluster.

2. Allow worker-specific pods to run on the **worker** node with:

```
kubectl label node <worker-node-name> nodeType=worker
```

3. If the node has status `Ready,SchedulingDisabled` run:

```
kubectl uncordon <node-name>
```

### Remove a worker node from the cluster

On the **master** node execute the following commands:

```
kubectl drain <node-name> --ignore-daemonsets
kubectl delete node <node-name>
```

## Firewall Configuration

Using firewalld the following rules should apply,

in the **master** node:

```
firewall-cmd --permanent --add-port=6443/tcp       # Kubelet api port
firewall-cmd --permanent --add-port=30000/tcp      # MIPEngine Controller port
```

on all nodes:

```
firewall-cmd --zone=public --permanent --add-rich-rule='rule protocol value="ipip" accept'  # Protocol "4" for "calico"-network-plugin.
```

These rules allow for kubectl to only be run on the **master** node.
