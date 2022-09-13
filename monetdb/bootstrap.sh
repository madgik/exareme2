#!/usr/bin/env bash
monetdbd get all /home/monetdb > /dev/null 2>&1
EXIT_STATUS=$?

if [ "$EXIT_STATUS" -eq 1 ]
then
  echo 'Initializing database'

  # Create the monetdb daemon
  monetdbd create $MONETDB_STORAGE
  monetdbd set port=50000 $MONETDB_STORAGE
  monetdbd set listenaddr=0.0.0.0  $MONETDB_STORAGE
  monetdbd start $MONETDB_STORAGE

  # Create the database instance
  monetdb create db
  monetdb set nclients=$MONETDB_NCLIENTS db
  monetdb set embedpy3=true db
  monetdb release db
  echo 'Database initialized'
else
  monetdbd start $MONETDB_STORAGE
  monetdb set nclients=$MONETDB_NCLIENTS db
  echo "Number of clients were set"
  monetdb start db
fi

tail -fn +1 $MONETDB_STORAGE/merovingian.log
