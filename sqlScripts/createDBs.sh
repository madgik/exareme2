monetdb stop  globaldb
monetdb destroy -f globaldb
monetdb stop  localdb1
monetdb destroy -f localdb1
monetdb stop  localdb2
monetdb destroy -f localdb2

monetdb create globaldb
monetdb set embedpy3=true globaldb
monetdb start globaldb

monetdb create localdb1
monetdb set embedpy3=true localdb1
monetdb start localdb1

monetdb create localdb2
monetdb set embedpy3=true localdb2
monetdb start localdb2

mclient -u monetdb -d globaldb ./globaldb_scr.sql
mclient -u monetdb -d localdb1 ./localdb1_scr.sql
mclient -u monetdb -d localdb2 ./localdb2_scr.sql
