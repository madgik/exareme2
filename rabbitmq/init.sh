#!/bin/sh

# Start configuration task on the background
( echo "Waiting for process to start..." ; \
sleep $RABBITMQ_SLEEP_BEFORE_CONFIGURATION ; \
echo "Configuring server ..." ; \
rabbitmqctl add_user $RABBITMQ_ADMIN_USER $RABBITMQ_ADMIN_PASSWORD ; \
rabbitmqctl add_vhost $RABBITMQ_ADMIN_VHOST ; \
rabbitmqctl set_user_tags $RABBITMQ_ADMIN_USER user_tag ; \
rabbitmqctl set_permissions -p $RABBITMQ_ADMIN_VHOST $RABBITMQ_ADMIN_USER '.*' '.*' '.*'  ) &

# $@ is used to pass arguments to the rabbitmq-server command.
# For example if you use it like this: docker run -d rabbitmq arg1 arg2,
# it will be as you run in the container rabbitmq-server arg1 arg2
rabbitmq-server $@
