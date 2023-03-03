# K6 with grafana and influxdb

## Deployment:

1. Start the containers:
   `docker-compose up -d influxdb grafana`
1. Make a test run of the script to prepare influxdb:
   `docker-compose run k6 run /scripts/run_random_algorithm.js`
1. Setup influxdb following this tutorial: https://k6.io/docs/results-output/real-time/influxdb-grafana/
1. Import this dashboard using influxdb as datasource: https://grafana.com/grafana/dashboards/2587
1. Run more algorithms:
   `docker-compose run k6 run /scripts/run_random_algorithm.js`
