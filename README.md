# MIP-Engine [![Maintainability](https://api.codeclimate.com/v1/badges/3e022eaf87579b79cf0f/maintainability)](https://codeclimate.com/github/madgik/MIP-Engine/maintainability) <a href="https://codeclimate.com/github/madgik/MIP-Engine/test_coverage"><img src="https://api.codeclimate.com/v1/badges/3e022eaf87579b79cf0f/test_coverage" /></a>

### Prerequisites

1. Install [python3.8](https://www.python.org/downloads/ "python3.8")

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

1. *Optional* To install tab completion for `invoke` run  (replacing `bash` with your shell)

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
   log_level = "INFO"
   framework_log_level ="INFO"
   monetdb_image = "madgik/mipenginedb:latest"
   rabbitmq_image = "madgik/mipengine_rabbitmq:latest"

   cdes_metadata_path = "./tests/demo_data"

   algorithm_folders = "./mipengine/algorithms,./tests/algorithms"

   node_registry_update_interval = 30

   [[nodes]]
   id = "globalnode"
   role = "GLOBALNODE"
   monetdb_port=50000
   rabbitmq_port=5670

   [[nodes]]
   id = "localnode1"
   role = "LOCALNODE"
   monetdb_port=50001
   rabbitmq_port=5671

   [[nodes]]
   id = "localnode2"
   role = "LOCALNODE"
   monetdb_port=50002
   rabbitmq_port=5672

   ```

   and then run the following command to create the config files that the node services will use

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
   inv attach --node <NODE-NAME>
   ```

1. Restart all the node/controller services and keep the same containers with

   ```
   inv start-node --all && inv start-controller --detached
   ```

#### Local Deployment (without single configuration file)

1. Create the node configuration files inside the `./configs/nodes/` directory following the `./mipengine/node/config.toml` template.

1. Install dependencies, start the containers and then the services with

   ```
   inv deploy --monetdb-image madgik/mipenginedb:dev1.2 --celery-log-level info
   ```

#### Start monitoring tools

1. Start Flower monitoring tool

   by choosing a specific node to monitor

   ```
   inv start-flower --node <NODE-NAME>
   ```

   or start a separate flower instance for all of the nodes with

   ```
   inv start-flower --all
   ```

   Then go to the respective address on your browser to start monitoring the nodes.

1. Kill all flower instances at any point with

   ```
   inv kill-flower
   ```

#### Algorithm Run

1. Make a post request, _e.g._
   ```
   python example_post_request.py
   ```
