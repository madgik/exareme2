## Monetdb 11.39.13 (Oct2020-SP3) dockerized

### Build
In order to change the initial monetdb configuration, go to the `bootstrap.sh`
inside the *config* folder.

To build a new image you must be on the project root `MIP-Engine/`, then
```
docker build -t <USERNAME>/mipenginedb:<IMAGETAG> -f monetdb_dockerized/Dockerfile .
```

## Run
Then run with
```
docker run -d -P -p 50000:50000 --name <CONTAINERNAME> <USERNAME>/mipenginedb:<IMAGETAG>
```

Access container db with
```
docker exec -it <CONTAINERNAME> mclient db
```
