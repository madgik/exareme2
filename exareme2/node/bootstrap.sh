#!/usr/bin/env bash
if [ -f "$CREDENTIALS_CONFIG_FOLDER/monetdb_password.sh" ]; then
    source $CREDENTIALS_CONFIG_FOLDER/monetdb_password.sh
fi
celery -A exareme2.node.node worker -l INFO --pool eventlet