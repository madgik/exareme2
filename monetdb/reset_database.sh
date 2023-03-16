#!/bin/bash
monetdbd stop /home/monetdb
rm -rf /home/monetdb/* /home/monetdb/.m*
./bootstrap.sh > /dev/null 2>&1 &
