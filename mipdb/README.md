## MIPDB container

### Usage

In order to load data into a node database, using kubernetes, you need to load them through the mipdb container.

You can enter the container through kubernetes and then use the mipdb commands.
For example:

```
mipdb --help
mipdb init --ip $DB_IP --port $PORT_IP
mipdb load-folder $DATA_PATH --ip $DB_IP --port $PORT_IP
```

The env variables you need to be aware of:

```
DATA_PATH = The path where the data models and datasets are.
DB_IP = The ip of the database.
PORT_IP = The port of the database.
```
