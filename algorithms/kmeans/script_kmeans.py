import random
import matplotlib.pyplot as plt
import numpy as np
from kmeans import _local,_global

num_of_datapoints_per_node_per_region=20

#3 Nodes
#Generate datapoints for the nodes
#For each node, random datapoints will be ghenerated in 3 regions
#region1: x in [10,15), y in [10,15)
r1_min_x=10;r1_max_x=55;
r1_min_y=10;r1_max_y=55;
#region2: x in [-10,-5), y in [10,15)
r2_min_x=-50;r2_max_x=-10;
r2_min_y=10;r2_max_y=55;
#region3: x in [-10,-5), y in [-10,-5)
r3_min_x=-50;r3_max_x=-10;
r3_min_y=-50;r3_max_y=-10;

colx_node1_region1=[random.uniform(r1_min_x,r1_max_x) for i in range(num_of_datapoints_per_node_per_region)]
colx_node1_region2=[random.uniform(r2_min_x,r2_max_x) for i in range(num_of_datapoints_per_node_per_region)]
colx_node1_region3=[random.uniform(r3_min_x,r3_max_x) for i in range(num_of_datapoints_per_node_per_region)]
colx_node1=colx_node1_region1+colx_node1_region2+colx_node1_region3
coly_node1_region1=[random.uniform(r1_min_y,r1_max_y) for i in range(num_of_datapoints_per_node_per_region)]
coly_node1_region2=[random.uniform(r2_min_y,r2_max_y) for i in range(num_of_datapoints_per_node_per_region)]
coly_node1_region3=[random.uniform(r3_min_y,r3_max_y) for i in range(num_of_datapoints_per_node_per_region)]
coly_node1=coly_node1_region1+coly_node1_region2+coly_node1_region3
node1_datapoints=np.array([colx_node1,coly_node1]).T

colx_node2_region1=[random.uniform(r1_min_x,r1_max_x) for i in range(num_of_datapoints_per_node_per_region)]
colx_node2_region2=[random.uniform(r2_min_x,r2_max_x) for i in range(num_of_datapoints_per_node_per_region)]
colx_node2_region3=[random.uniform(r3_min_x,r3_max_x) for i in range(num_of_datapoints_per_node_per_region)]
colx_node2=colx_node2_region1+colx_node2_region2+colx_node2_region3
coly_node2_region1=[random.uniform(r1_min_y,r1_max_y) for i in range(num_of_datapoints_per_node_per_region)]
coly_node2_region2=[random.uniform(r2_min_y,r2_max_y) for i in range(num_of_datapoints_per_node_per_region)]
coly_node2_region3=[random.uniform(r3_min_y,r3_max_y) for i in range(num_of_datapoints_per_node_per_region)]
coly_node2=coly_node2_region1+coly_node2_region2+coly_node2_region3
node2_datapoints=np.array([colx_node2,coly_node2]).T

colx_node3_region1=[random.uniform(r1_min_x,r1_max_x) for i in range(num_of_datapoints_per_node_per_region)]
colx_node3_region2=[random.uniform(r2_min_x,r2_max_x) for i in range(num_of_datapoints_per_node_per_region)]
colx_node3_region3=[random.uniform(r3_min_x,r3_max_x) for i in range(num_of_datapoints_per_node_per_region)]
colx_node3=colx_node3_region1+colx_node3_region2+colx_node3_region3
coly_node3_region1=[random.uniform(r1_min_y,r1_max_y) for i in range(num_of_datapoints_per_node_per_region)]
coly_node3_region2=[random.uniform(r2_min_y,r2_max_y) for i in range(num_of_datapoints_per_node_per_region)]
coly_node3_region3=[random.uniform(r3_min_y,r3_max_y) for i in range(num_of_datapoints_per_node_per_region)]
coly_node3=coly_node3_region1+coly_node3_region2+coly_node3_region3
node3_datapoints=np.array([colx_node3,coly_node3]).T

#print(f"colx_node1->{colx_node1}\n")
#print(f"colx_node2->{colx_node2}\n")
#print(f"colx_node3->{colx_node3}\n")

num_of_clusters=3
initial_centroids=_global(num_of_clusters,[])
print(f"initial centroids->{initial_centroids}")

num_of_iterations=10
centroids=initial_centroids
for i in range(num_of_iterations):
    new_centroids_node1=_local(centroids,node1_datapoints)
    new_centroids_node2=_local(centroids,node1_datapoints)
    new_centroids_node3=_local(centroids,node1_datapoints)
    centroids=_global(num_of_clusters,[new_centroids_node1,new_centroids_node2,new_centroids_node3])
print(f"FINAL centroids->{centroids}")

#plottinng--------------------------------------------------------------------------------------

plt.plot(node1_datapoints[:,0],node1_datapoints[:,1],'cd',label="node1 datapoints");
plt.plot(node2_datapoints[:,0],node2_datapoints[:,1],'ms',label="node2 datapoints");
plt.plot(node3_datapoints[:,0],node3_datapoints[:,1],'yo',label="node3 datapoints");

c=np.array(initial_centroids)
plt.plot(c[:,0],c[:,1],'rx',markersize=10,label="initial centroids")

c=np.array(centroids)
plt.plot(c[:,0],c[:,1],'gx',markersize=12,label="final centroids")
plt.legend(loc='lower right')
plt.show()

