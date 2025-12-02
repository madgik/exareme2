from lazy_aggregation import build_dependency_graph
from lazy_aggregation import visualize_graph

from exaflow.algorithms.exareme3.library.stats.stats import pca

nodes, edges = build_dependency_graph(pca)

for node in nodes:
    print(node)
for edge in edges:
    print(edge)
# visualize_graph(nodes, edges, "pca_graph")
