## Configuration with microk8s

The following packages need to be installed on **master/worker** nodes:

```
microk8s (sudo snap install micro8s --classic)
```

Additional configuration needed on the **master** node only:

```
microk8s enable dns ingress registry helm3
```

(Optional) You can enable the k8s dashboard with:

```
microk8s enable dashboard
```

### Hostname configuration

Currently, each node is distinguished from another one using their hostnames and they <b>MUST</b> be alphanumeric.
<br>If the node's hostname does not comply with that convention it has to be configured
from the kubelet following this [guide](https://kubernetes.io/docs/reference/labels-annotations-taints/#kubernetesiohostname).

To configure kubelet in microk8s you can do the following, quoting from https://microk8s.io/docs/troubleshooting:

```
To fix this you can change the hostname or use the --hostname-override argument
in kubelet’s configuration in /var/snap/microk8s/current/args/kubelet.
```

## Cluster Management

### Configure the master node to run pods

Allow master-specific pods to run on the **master** node with:

```
kubectl taint nodes --all node-role.kubernetes.io/master-
kubectl label node <master-node-name> master=true
```

### Add a worker node to the cluster

1. On the **master** node, get the join token with the following command:

```
microk8s add-node
```

Use the provided command, including `--worker` on the **worker** node to join the cluster.

2. Allow worker-specific pods to run on the **worker** node with:

```
microk8s kubectl label node <worker-node-name> worker=true
```

3. If the node has status `Ready,SchedulingDisabled` run:

```
microk8s kubectl uncordon <node-name>
```

### Remove a worker node from the cluster

On the **master** node execute the following commands:

```
microk8s kubectl drain <node-name> --ignore-daemonsets
microk8s kubectl delete node <node-name>
```

On the **worker** node execute the following command:

```
microk8s leave
```

### (Optional) Configure SMPC workers in the cluster

The SMPC cluster requires 3 different nodes, to be used as "players", in order to secure the computation.

If SMPC is enabled in the deployment use the following command on the **master** or **worker** nodes (3 nodes are needed):

```
microk8s kubectl label node <node-name> smpc_player=true
```

## Firewall Configuration

Using firewalld the following rules should apply,

in the **master** node, to expose the controller api, if it's used outside the cluster:

```
firewall-cmd --permanent --add-port=30000/tcp      # MIPEngine Controller port
```

on all nodes:

```
sudo firewall-cmd --permanent --add-port={10255,12379,25000,16443,10250,10257,10259,32000}/tcp
sudo firewall-cmd --reload
```

If there are node to node communication problems we also need to add on all nodes:

```
SUBNET=`cat /var/snap/microk8s/current/args/cni-network/cni.yaml | grep CALICO_IPV4POOL_CIDR -a1 | tail -n1 | grep -oP '[\d\./]+'`
sudo firewall-cmd --permanent --new-zone=microk8s-cluster
sudo firewall-cmd --permanent --zone=microk8s-cluster --set-target=ACCEPT
sudo firewall-cmd --permanent --zone=microk8s-cluster --add-source=$SUBNET
sudo firewall-cmd --reload
```
