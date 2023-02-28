## MIPDB container

### Usage

In order to load data into a node database, using kubernetes, you need to load them through the mipdb container.

You can enter the container through kubernetes and then use the mipdb commands.
For example:

```
mipdb --help
mipdb init
mipdb load-folder $DATA_PATH
```

The env variables you need to be aware of:

```
DATA_PATH = The path where the data models and datasets are.
DB_IP=The ip of the database.
DB_PORT=The port of the database.
DB_USERNAME=The username of the database.
DB_PASSWORD=The password of the database.
DB_NAME=The name of the database.
```
