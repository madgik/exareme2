#!/bin/bash

# Check if user exists and can be authenticated
if ! rabbitmqctl authenticate_user $RABBITMQ_ADMIN_USER $RABBITMQ_ADMIN_PASSWORD > /dev/null 2>&1
then
  exit 1
fi

# Check if vhost exists
if ! rabbitmqctl list_permissions -p $RABBITMQ_ADMIN_VHOST > /dev/null 2>&1
then
  exit 1
fi
