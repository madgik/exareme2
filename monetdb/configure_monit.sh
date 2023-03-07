#!/bin/bash
cat << EOF > /etc/monit/monitrc
# Monit configurations
  set log $MONIT_CONFIG_FOLDER/monit.log
  set idfile $MONIT_CONFIG_FOLDER/id
  set statefile $MONIT_CONFIG_FOLDER/state

# Monit cycle time
  set daemon 1 #The amount of seconds for each cycle

# Monit monetdb monitor
  check process monetdb matching mserver
    start program = "/bin/bash -c '/usr/local/bin/monetdb/bin/monetdb release db && /usr/local/bin/monetdb/bin/monetdb start db'"
    stop program  = "/bin/bash -c '/usr/local/bin/monetdb/bin/monetdb lock db && /usr/local/bin/monetdb/bin/monetdb stop db'"
    if totalmem > $SOFT_RESTART_MEMORY_LIMIT MB for 10 cycles then restart
    if totalmem > $HARD_RESTART_MEMORY_LIMIT MB for 1 cycles then restart
EOF

monit
monit reload
