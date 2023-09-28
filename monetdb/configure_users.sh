#!/bin/bash
  <<comments
ADMIN:
  This user will be responsible for the loading and the modification of the metadata of the data models (this role exists only for mipdb).
  This user will be able to access all tables of the database.
LOCAL_USER:
  This user will be responsible for the creation of the tables, functions as well as, the execution of udfs that are required to run an algorithm.
  This user will be able to access the tables of the database that were created through their account.
  User 'admin' has granted to the 'local' user, the ability to retrieve the data from tables created by the mipdb (datasets,data models, primary_data, etc).
PUBLIC_USER:
  This user will ONLY be able to retrieve some specific 'public' tables.
  This role mainly controls remote table creation between databases.


USERS PASSWORD:
  Only 'public' user will have a password that will be the same in all the databases so all the databases can access each others' public tables
  'public' user and 'admin' will share the same password, on the production environment this password will differ for each database.
  For the development environment the passwords are the following:
    USER    | PASSWORD
    admin   | admin
    executor| executor
    guest   | guest
comments

# The script has 2 execution options:
# the first one is used upon database creation.
# the second one is used upon database restart.
CREATION_CONFIGURATION_ARG="creation"
RESTART_CONFIGURATION_ARG="restart"

if [ -f "$CREDENTIALS_CONFIG_FOLDER/monetdb_password.sh" ]; then
    source $CREDENTIALS_CONFIG_FOLDER/monetdb_password.sh
fi

if [[ $1 = "$CREATION_CONFIGURATION_ARG" ]] ; then
  #######################################################
  # Set default username/password as default user
  #######################################################
  echo "user=monetdb" > /home/.monetdb
  echo "password=monetdb" >> /home/.monetdb

  #######################################################
  # Create all users
  #######################################################
  mclient db -s "CREATE USER $MONETDB_LOCAL_USERNAME WITH PASSWORD '$MONETDB_LOCAL_PASSWORD' NAME '$MONETDB_LOCAL_USERNAME';"
  echo "User '$MONETDB_LOCAL_USERNAME' created."
  mclient db -s "CREATE USER $MONETDB_PUBLIC_USERNAME WITH PASSWORD '$MONETDB_PUBLIC_PASSWORD' NAME '$MONETDB_PUBLIC_USERNAME';"
  echo "User '$MONETDB_PUBLIC_USERNAME' created."
  mclient db -s "ALTER USER SET PASSWORD '$MONETDB_LOCAL_PASSWORD' USING OLD PASSWORD 'monetdb'; ALTER USER \"monetdb\" RENAME TO \"$MONETDB_ADMIN_USERNAME\";"
  echo "Renamed 'monetdb' master user to '$MONETDB_ADMIN_USERNAME'."

  #######################################################
  # Set LOCAL_USER username/password as default user
  #######################################################
  echo "user=$MONETDB_LOCAL_USERNAME" > /home/.monetdb
  echo "password=$MONETDB_LOCAL_PASSWORD" >> /home/.monetdb

elif [[ $1 = "$RESTART_CONFIGURATION_ARG" ]]; then
  #######################################################
  # Set LOCAL_USER username/password as default user
  #######################################################
  echo "user=$MONETDB_LOCAL_USERNAME" > /home/.monetdb
  echo "password=$MONETDB_LOCAL_PASSWORD" >> /home/.monetdb

else
  echo "Invalid argument provided. '$CREATION_CONFIGURATION_ARG' or '$RESTART_CONFIGURATION_ARG' is allowed, given: '$1'"
  exit 1

fi
