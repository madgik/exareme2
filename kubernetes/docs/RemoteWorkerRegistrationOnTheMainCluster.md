## Main Cluster Registration for Remote MicroK8s Worker

This document contains **only** the steps that must be executed on the **managed main cluster** after the remote MicroK8s worker service has been exported via Submariner.

> **Prerequisites**
>
> - Remote worker setup is complete (see *Remote MicroK8s Worker Setup* guide).
>
> - Submariner connectivity between clusters is healthy and the worker service has been exported:
>
>   ```bash
>   subctl export service exareme2-remote-worker-service --namespace <namespace>
>   ```
>
> **Service-name disclaimer**
> By default the Helm chart creates a headless Service named `exareme2-remote-worker-service`.
> If you will register **more than one** remote worker cluster,
> you **must** give each headless Service a **unique** name (for example `exareme2-<node>-worker-service`) **before** exporting it with Submariner;
> Otherwise the resulting `ServiceImport` objects will collide on the main cluster and only the first worker will be reachable.

______________________________________________________________________

### 1  Validate the ServiceImport

Check that Submariner created a `ServiceImport` resource for the remote worker:

```bash
kubectl -n <namespace> get serviceimport \
  exareme2-<node-name>-worker-service
```

A healthy `ServiceImport` should show `type=ClusterSetIP` and a populated endpoint.

______________________________________________________________________

### 2  Add the Worker DNS Record

Edit the Exareme2 Controller values file (example path `mip-infrastructure/customizations/exareme2-values.yaml`) and append the new DNS entry:

```yaml
controller:
  workers_dns: >
    exareme2-workers-service.<namespace>.svc.cluster.local,
    exareme2-remote-worker-service.<namespace>.svc.clusterset.local
```

Commit the change if you are using GitOps, or apply it manually with `helm upgrade`.

______________________________________________________________________

### 3  Force the Controller to Refresh (optional)

The controller will pick up the new DNS entry automatically within ~5 minutes. To refresh immediately:

```bash
kubectl delete pod -n <namespace> -l app=exareme2-controller
```

A new controller pod starts and discovers the remote worker.

______________________________________________________________________

### Federation Workflow Recap (for reference)

1. **ServiceExport** – Created on the remote node.
1. **ServiceImport** – Auto‑generated on the main cluster by Submariner.
1. **DNS Propagation** – Cluster‑set DNS entry becomes resolvable.
1. **Controller Discovery** – Exareme2 Controller reads `workers_dns` and begins sending jobs to the remote worker.

With this, the remote worker is fully integrated into your Exareme2 federation.
