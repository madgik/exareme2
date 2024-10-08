FROM rabbitmq:3.9.29-management

ENV RABBITMQ_ADMIN_USER user
ENV RABBITMQ_ADMIN_PASSWORD password
ENV RABBITMQ_ADMIN_VHOST user_vhost

ADD rabbitmq/init.sh /init.sh
ADD rabbitmq/configure.sh /configure.sh
ADD rabbitmq/healthcheck.sh /healthcheck.sh
RUN chmod +x /init.sh
RUN chmod +x /configure.sh
RUN chmod +x /healthcheck.sh
EXPOSE 5672

HEALTHCHECK CMD bash /healthcheck.sh

CMD ["/init.sh"]
