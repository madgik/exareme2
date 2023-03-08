#!/bin/bash
  <<comments
ADMIN:
  This user will be responsible for the loading and the modification of the metadata of the data models (this role exists only for mipdb).
  This user will be able to access all tables of the database.
EXECUTOR:
  This user will be responsible for the creation of the tables, functions as well as, the execution of udfs that are required to run an algorithm.
  This user will be able to access the tables of the database that were created through their account.
  User 'admin' has granted to the 'executor', the ability to retrieve the data from tables created by the mipdb (datasets,data models, primary_data, etc).
GUEST:
  This user will ONLY be able to retrieve some specific 'public' tables.
  This role mainly controls remote table creation between databases.


USERS PASSWORD:
  Only user 'guest' will have a hard-coded password that will be the same in all the databases so all the databases can access each others' public tables
  'executor' and 'admin' password on the production environment should have different password for each database.
  For the development environment the passwords are the following:
    USER    | PASSWORD
    admin   | admin
    executor| executor
    guest   | guest
comments


mclient db -s "CREATE USER executor WITH PASSWORD '$EXECUTOR_PASSWORD' NAME 'executor';"
echo "User 'executor' created."
mclient db -s "CREATE USER guest WITH PASSWORD '$GUEST_PASSWORD' NAME 'guest';"
echo "User 'guest' created."
mclient db -s "ALTER USER SET PASSWORD '$ADMIN_PASSWORD' USING OLD PASSWORD 'monetdb'; ALTER USER \"monetdb\" RENAME TO \"admin\";"
echo "Renamed 'monetdb' master user to 'admin'."
echo "user=executor" > /home/.monetdb
echo "password=$EXECUTOR_PASSWORD" >> /home/.monetdb