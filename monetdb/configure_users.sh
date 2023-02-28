#!/bin/bash

mclient db -s "CREATE USER executor WITH PASSWORD '$EXECUTOR_PASSWORD' NAME 'executor';"
echo "User 'executor' created."
mclient db -s "CREATE USER guest WITH PASSWORD '$GUEST_PASSWORD' NAME 'guest';"
echo "User 'guest' created."
mclient db -s "ALTER USER SET PASSWORD '$ADMIN_PASSWORD' USING OLD PASSWORD 'monetdb'; ALTER USER \"monetdb\" RENAME TO \"admin\";"
echo "Renamed 'monetdb' master user to 'admin'."
echo "user=executor" > /home/.monetdb
echo "password=$EXECUTOR_PASSWORD" >> /home/.monetdb
