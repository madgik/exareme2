#!/bin/bash

while ! rabbitmqctl list_users  > /dev/null 2>&1
do
  echo "Waiting for process to start..."
  sleep 2
done

echo "Rabbitmq started, configuring server ..."
rabbitmqctl add_user $RABBITMQ_ADMIN_USER $RABBITMQ_ADMIN_PASSWORD
rabbitmqctl set_user_tags $RABBITMQ_ADMIN_USER user_tag administrator
rabbitmqctl add_vhost $RABBITMQ_ADMIN_VHOST
rabbitmqctl set_permissions -p $RABBITMQ_ADMIN_VHOST $RABBITMQ_ADMIN_USER '.*' '.*' '.*'
