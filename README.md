# Exareme2 [![Maintainability](https://qlty.sh/gh/madgik/projects/MIP-Engine/maintainability.svg)](https://qlty.sh/gh/madgik/projects/MIP-Engine) [![Code Coverage](https://qlty.sh/gh/madgik/projects/MIP-Engine/coverage.svg)](https://qlty.sh/gh/madgik/projects/MIP-Engine)

### Prerequisites

1. Install [python3.10](https://www.python.org/downloads/ "python3.10")

1. Install [poetry](https://python-poetry.org/ "poetry")
   It is important to install `poetry` in isolation, so follow the
   recommended installation method.

## Setup

#### Environment Setup

1. Install dependencies

   ```
   poetry install
   ```

1. Activate virtual environment

   ```
   poetry shell
   ```

1. *Optional* To install tab completion for `invoke` run (replacing `bash` with your shell)

   ```
   source <(poetry run inv --print-completion-script bash)
   ```

1. _Optional_ `pre-commit` is included in development dependencies. To install hooks

   ```
   pre-commit install
   ```

#### Local Deployment

1. Create a deployment configuration file `.deployment.toml` using the following template

   ```
   ip = "172.17.0.1"
   federation = "dementia"
   log_level = "DEBUG"
   framework_log_level ="INFO"
   monetdb_image = "madgik/exareme2_db:dev"
   rabbitmq_image = "madgik/exareme2_rabbitmq:dev"

   monetdb_nclients = 128
   monetdb_memory_limit = 2048 # MB

   exareme2_algorithm_folders = "./exareme2/algorithms/exareme2,./tests/algorithms/exareme2"
   flower_algorithm_folders = "./exareme2/algorithms/flower,./tests/algorithms/flower"
   exaflow_algorithm_folders = "./exareme2/algorithms/exaflow,./tests/algorithms/exaflow"

   worker_landscape_aggregator_update_interval = 30
   celery_tasks_timeout = 20
   celery_cleanup_task_timeout=2
   celery_run_udf_task_timeout = 120

   [flower]
   enabled = true
   execution_timeout = 60
   server_port = 8080

   [controller]
   port = 5000

   [privacy]
   minimum_row_count = 10
   protect_local_data = false

   [cleanup]
   workers_cleanup_interval=10
   contextid_release_timelimit=3600 #an hour

   [aggregation_server]
   enabled = true
   port = 50051
   max_grpc_connections = 10
   max_wait_for_aggregation_inputs = 10

   [monetdb]
   enabled = true

   [smpc]
   enabled=false
   optional=false
   get_result_interval = 10
   get_result_max_retries = 100
   smpc_image="gpikra/coordinator:v7.0.7.4"
   db_image="mongo:5.0.8"
   queue_image="redis:alpine3.15"
   [smpc.dp]
   enabled = false
   # sensitivity = 1
   # privacy_budget = 0.1

   [[workers]]
   id = "globalworker"
   role = "GLOBALWORKER"
   rabbitmq_port=5670
   monetdb_port=50000
   monetdb_password="executor"
   local_monetdb_username="executor"
   local_monetdb_password="executor"
   public_monetdb_username="guest"
   public_monetdb_password="guest"

   [[workers]]
   id = "localworker1"
   role = "LOCALWORKER"
   rabbitmq_port=5671
   monetdb_port=50001
   local_monetdb_username="executor"
   local_monetdb_password="executor"
   public_monetdb_username="guest"
   public_monetdb_password="guest"
   smpc_client_port=9001

   [[workers]]
   id = "localworker2"
   role = "LOCALWORKER"
   rabbitmq_port=5672
   monetdb_port=50002
   local_monetdb_username="executor"
   local_monetdb_password="executor"
   public_monetdb_username="guest"
   public_monetdb_password="guest"
   smpc_client_port=9002

   ```

   and then run the following command to create the config files that the worker services will use

   ```
   inv create-configs
   ```

1. Install dependencies, start the containers and then the services with

   ```
   inv deploy
   ```

1. _Optional_ Load the data into the db with
   (It is compulsory if you want to run an algorithm)

   ```
   inv load-data
   ```

1. Attach to some service's stdout/stderr with

   ```
   inv attach --controller
   ```

   or

   ```
   inv attach --worker <WORKER-NAME>
   ```

1. Restart all the worker/controller services and keep the same containers with

   ```
   inv start-worker --all && inv start-controller --detached
   ```

#### Local Deployment (without single configuration file)

1. Create the worker configuration files inside the `./configs/workers/` directory following the `./exareme2/worker/config.toml` template.

1. Install dependencies, start the containers and then the services with

   ```
   inv deploy --monetdb-image madgik/exareme2_db:dev1.2 --celery-log-level info
   ```

#### Start monitoring tools

1. Start Flower monitoring tool

   by choosing a specific worker to monitor

   ```
   inv start-flower --worker <WORKER-NAME>
   ```

   or start a separate flower instance for all of the workers with

   ```
   inv start-flower --all
   ```

   Then go to the respective address on your browser to start monitoring the workers.

1. Kill all flower instances at any point with

   ```
   inv kill-flower
   ```

#### Execute an algorithm

- Examples
  ```
  ./run_algorithm -a pca -y leftamygdala lefthippocampus -d ppmi0 -m dementia:0.1
  ```
  ```
  ./run_algorithm -a pearson -y leftamygdala lefthippocampus -d ppmi0 -m dementia:0.1 -p alpha 0.95
  ```

# Acknowledgement

This project/research received funding from the European Union’s Horizon 2020 Framework Programme for Research and Innovation under the Framework Partnership Agreement No. 650003 (HBP FPA).
