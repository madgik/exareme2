#!/usr/bin/env bash

# Create the monetdb daemon
monetdbd create $MONETDB_STORAGE
monetdbd set port=50000 $MONETDB_STORAGE
monetdbd set listenaddr=0.0.0.0  $MONETDB_STORAGE
monetdbd start $MONETDB_STORAGE

# Create the database instance
monetdb create db
monetdb set embedpy3=true db
monetdb set max_clients=128 db
monetdb release db

tail -fn +1 $MONETDB_STORAGE/merovingian.log
