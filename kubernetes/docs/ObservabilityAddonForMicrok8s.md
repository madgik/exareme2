## Prometheus and Loki Installation on Microk8s

### Addon installation

Both the Prometheus and Loki (Log Search) are packaged inside the `observability` addon of microk8s. In order to install run the following command after setting up the microk8s cluster:

```
microk8s enable observability
```

### Exposing Grafana and Prometheus

In order to expose grafana you need the following command:

```
microk8s kubectl port-forward -n observability service/kube-prom-stack-grafana --address 0.0.0.0 9091:80 &
```

In order to login to grafana, use the password provided after installation.

In order to expose prometheus:

```
microk8s kubectl port-forward -n observability service/prometheus-operated --address 0.0.0.0 9090:9090 &
```

### Setting up Grafana for Node Resources Utilization

In order to see the system resources used by your pods, after you login to grafana go to:

```
Dashboards -> Browse -> Kubernetes / Compute Resources / Node (Pods)
```

You can also use any other dashboard you prefer, for other visualizations, or make your own!

### Setting up Log Search

We are going to use Loki to perform log search across the federation, go to:

```
Explore ->
  At the top of the page select Loki ->
    As label filters add "Namespace" = "default" (that will search only in exareme containers) ->
      In the "Line Contains" field add the text to search for. ->
        Logs from all containers will be aggregated at the bottom.
```

Example:

![image](https://github.com/madgik/exaflow/assets/15667989/6e6e28bc-0363-4e33-b8a1-07ad87ef368c)

### Notes

As stated after the addon installation:

```
Note: the observability stack is setup to monitor only the current nodes of the MicroK8s cluster.
For any nodes joining the cluster at a later stage this addon will need to be set up again.
```
