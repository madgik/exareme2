FROM rabbitmq:3.8

ENV RABBITMQ_ADMIN_USER user
ENV RABBITMQ_ADMIN_PASSWORD password
ENV RABBITMQ_ADMIN_VHOST user_vhost
ENV RABBITMQ_SLEEP_BEFORE_CONFIGURATION 30

ADD rabbitmq/init.sh /init.sh
RUN chmod +x /init.sh
EXPOSE 5672



CMD ["/init.sh"]
