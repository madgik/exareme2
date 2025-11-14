#!/usr/bin/env bash
echo 'Monetdb bootstrap script started.'

monetdbd get all /home/monetdb > /dev/null 2>&1
EXIT_STATUS=$?

if [ "$EXIT_STATUS" -eq 1 ]
then
  echo 'Initializing the database...'

  # Create the monetdb daemon
  chmod -R 777 $MONETDB_STORAGE
  monetdbd create $MONETDB_STORAGE
  monetdbd set port=50000 $MONETDB_STORAGE
  monetdbd set listenaddr=0.0.0.0  $MONETDB_STORAGE
  monetdbd start $MONETDB_STORAGE

  # Create the database instance
  monetdb create db
  monetdb set nclients=$MONETDB_NCLIENTS db
  monetdb set vmmaxsize=$MAX_MEMORY db
  monetdb set memmaxsize=$MAX_MEMORY db
  monetdb set embedc=true db
  # 'monetdb set mal_for_all=yes db' is needed to be able to access tables on a remote database with user that is not the master user.
  monetdb set mal_for_all=yes db
  monetdb release db
  monetdb start db

  echo 'Database initialized.'

  ./configure_users.sh creation

else
  echo 'Checking if previous instances are still running (from other containers).'
  monetdbd_stopped_status_message="no monetdbd is serving this dbfarm"
  while [[ "$(monetdbd get status /home/monetdb)" != *$monetdbd_stopped_status_message* ]]
  do
      echo 'Waiting for previous monetdbd instance to stop...'
      ((retries++)) && ((retries==60)) && echo 'Previous instance didnt stop, exiting.' && exit 1   # Check 60 times if no daemon is already running, then exit.
      sleep 1
  done
  echo 'No monetdbd instances are running.'

  echo 'Starting the already existing database...'
  chmod -R 777 $MONETDB_STORAGE
  monetdbd start $MONETDB_STORAGE
  monetdb set vmmaxsize=$MAX_MEMORY db
  monetdb set memmaxsize=$MAX_MEMORY db
  monetdb set nclients=$MONETDB_NCLIENTS db
  monetdb set mal_for_all=yes db
  monetdb start db
  echo 'Database restarted.'

  ./configure_users.sh restart

fi

./configure_monit.sh
