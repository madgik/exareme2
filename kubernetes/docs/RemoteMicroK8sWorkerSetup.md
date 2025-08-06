## Remote MicroK8s Worker Setup

This guide covers **only** the actions you need to perform on the **remote node** that will run MicroK8s and act as an Exareme2 worker.

> **Prerequisites**
>
> - The remote node runs Ubuntu 20.04/22.04 (or compatible) with outbound internet access.
> - **MicroK8s is already installed and running** on the node (`microk8s status --wait-ready`).
> - Submariner connectivity between this MicroK8s cluster and the **managed main cluster** is already established and healthy (`subctl show all`).
> - You have `sudo` access on the remote node.
>
> All commands below assume **bash**. Prepend `sudo` where necessary.

______________________________________________________________________

### 1  Enable Required Add‑ons

Only the following add‑ons are required for a remote Exareme2 worker:

```bash
microk8s enable helm3
```

Verify:

```bash
microk8s kubectl get pods -A
```

Proceed when all add‑on pods are **Running** or **Completed**.

> **Note**  The `ingress` add‑on is **not** required on the remote worker node.

______________________________________________________________________

### 2  Label the Node

```bash
microk8s kubectl label node $(hostname) worker=true
```

______________________________________________________________________

### 3  Deploy Exareme2 Worker via Helm

Clone (or copy) the Exareme2 Helm chart and run:

```bash
cd exareme2/kubernetes
git checkout remote-node
microk8s helm3 install exareme2-worker .
```

Monitor:

```bash
watch -n2 'microk8s kubectl get pods -A'
```

Wait until all Exareme2 pods show **Running** or **Completed**.

______________________________________________________________________

### 4  Initialise the MIPDB Database

```bash
POD=$(microk8s kubectl get pods -A -o name | grep db-importer | head -1)
microk8s kubectl exec -it $POD -c db-importer -- mipdb init
```

______________________________________________________________________

### 5  Load Data into the Worker

1. Copy your CSV folders into **/opt/data** *inside* the importer container, or mount them via the host path declared in your `values.yaml` (default `/opt/exareme2/localworker/csvs`).

1. Import the data:

   ```bash
   microk8s kubectl exec -it $POD -c db-importer -- mipdb load-folder /opt/data
   ```

______________________________________________________________________

### 6  Export the Worker Service (Submariner)

Once the importer pod is healthy and data are loaded, expose the worker service so the main cluster can discover it.

> **Service‑name disclaimer**
> By default the Helm chart creates a headless Service named `exareme2-remote-worker-service`.
> If you will register **more than one** remote worker cluster,
> you **must** give each headless Service a **unique** name (for example `exareme2-<node>-worker-service`) **before** exporting it with Submariner;
> Otherwise the resulting `ServiceImport` objects will collide on the main cluster and only the first worker will be reachable.

```bash
# Find the service name
echo "Worker Services:" && microk8s kubectl get svc -n <namespace> | grep worker

# Export the service
authNamespace=<namespace>
svcName=exareme2-remote-worker-service
subctl export service $svcName --namespace $authNamespace
```
