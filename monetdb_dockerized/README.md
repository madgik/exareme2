## Monetdb 11.37.11 (June-SP1) dockerized

### Build
In order to change the initial monetdb configuration, go to the `bootstrap.sh`
inside the *config* folder.

To build a new image you must be on the project root `MIP-Engine/`, then
```
docker build -t <USERNAME>/<IMAGENAME>:<IMAGETAG> -f monetdb_dockerized/Dockerfile .
```

## Run
Then run with
```
docker run -d -P -p 50000:50000 --name <CONTAINERNAME> <USERNAME>/<IMAGENAME>:<IMAGETAG>
```

Access container db with
```
docker exec -it <CONTAINERNAME> mclient db
```
