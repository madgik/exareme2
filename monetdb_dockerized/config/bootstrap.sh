#!/usr/bin/env bash

# Create the monetdb daemon
monetdbd create /home/monetdb
monetdbd set port=50000 /home/monetdb
monetdbd set listenaddr=0.0.0.0  /home/monetdb
monetdbd start /home/monetdb

# Create the database instance
monetdb create db
monetdb set embedpy3=true db
monetdb release db

tail -fn +1 /home/monetdb/merovingian.log
