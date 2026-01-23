variable "REGISTRY" {
  default = "localhost:5001"
}

group "default" {
  targets = ["worker", "controller", "aggregation_server"]
}

target "worker" {
  context = "."
  dockerfile = "exaflow/worker/Dockerfile"
  tags = ["${REGISTRY}/madgik/exaflow_worker:dev"]
}

target "controller" {
  context = "."
  dockerfile = "exaflow/controller/Dockerfile"
  tags = ["${REGISTRY}/madgik/exaflow_controller:dev"]
}

target "aggregation_server" {
  context = "."
  dockerfile = "aggregation_server/Dockerfile"
  tags = ["${REGISTRY}/madgik/exaflow_aggregation_server:dev"]
}
