import numpy as np
from itertools import chain

def _local(centroids,column_x,column_y):
    new_centroids=kmeans_one_step(centroids,column_x,column_y)
    return new_centroids

def _global(num_of_clusters,local_centroids=[]):
    #when kmeans starts the centroids (the centers of the clusters) will be randomly generated
    if not local_centroids:
        new_centroids=[]
        for i in range(num_of_clusters):
            new_centroids.append([np.random.randint(100),np.random.randint(100)])
        return new_centroids

    #The calculated centroids from the local nodes have to be averaged out, in order to have a single of centroids to continue with the algorithm
    #Nevertheless, they have to be averaged out in some structured way in order to end up with a new set of centroids and not just one centroid as the result of 
    #all local centroids averaged together. The approach here is to handle one of the centroids sets returned by a node as the refference set of centroids and apply again
    #one step of the kmeans clustering on it, with the datapoints beeing the centroids returned from the other nodes. 
    num_of_centroids_returned_per_node=[ len(centroids) for centroids in local_centroids]

    #...one of the centroids sets (returned by the local step of the algorithm) with the most centroids is chosen as the refference centroids set 
    m=max(num_of_centroids_returned_per_node)
    index_of_node_with_max_number_of_centroids=num_of_centroids_returned_per_node.index(max(num_of_centroids_returned_per_node))
    centroids=local_centroids[index_of_node_with_max_number_of_centroids]

    #...with the datapoints beeing the centroids returned from the other nodes
    local_centroids.pop(index_of_node_with_max_number_of_centroids)
    data_points=local_centroids

    flatten=list(chain.from_iterable(data_points))
    colx=[x[0] for x in flatten]
    coly=[x[1] for x in flatten]

    #... apply again one step of the kmeans clustering on it, with the datapoints beeing the centroids returned from the other nodes. 
    new_centroids=kmeans_one_step(centroids,colx,coly)

    #The kmeans_one_step MIGHT return less number of centroids than the length of the centroids which it was called with, when zero datapoints are assigned to certain centroids(clusters)
    #In that case we generate a random new centroid to proceed with the algorithm
    if len(new_centroids)<num_of_clusters:
        num_of_centroids_missing=num_of_clusters-len(new_centroids)
        min_x=min(colx)
        max_x=max(colx)
        max_y=max(coly)
        min_y=min(coly)
        for i in range(num_of_centroids_missing):
            if min_x!=max_x:
                x_random=np.random.uniform(min_x,max_x)
            else:
                x_random=np.random.uniform()
            if min_y!=max_y:
                y_random=np.random.uniform(min_y,max_y)
            else:
                y_random=np.random.uniform()
            new_centroids.append([x_random,y_random])
            
    return new_centroids

#NOTE: The kmeans_one_step MIGHT return less number of centroids than the length of the centroids which it was called with, when zero datapoints are assigned to certain centroids(clusters)
def kmeans_one_step(centroids,column_x,column_y):
    #calculate distance between all datapoints(centroids,column_x,column_y) to all centroids
    distances=[]
    data_points=np.array([column_x,column_y]).T
    for data_point in data_points:
        dist_row=[]
        for centroid in centroids:
            dist=abs(np.linalg.norm(data_point-centroid))
            dist_row.append(dist)
        distances.append(dist_row)

    #assign each datapoint to the closest centroid (assign to clusters)
    cluster_assignment=[] 
    for data_point_distances in distances:
        min_index=np.where(data_point_distances==np.amin(data_point_distances))
        cluster_assignment.append(min_index[0][0])

    #calculate new centroids by calculatingt the mean of the datapoints assigned to each cluster
    new_centroids={k:0 for k in range(len(centroids))} 
    number_of_times_encountered={k:0 for k in range(len(new_centroids))}
    for i in range(len(cluster_assignment)):
        new_centroids[cluster_assignment[i]]+=data_points[i]
        number_of_times_encountered[cluster_assignment[i]]+=1
    #averaging them..
    for i in range(len(new_centroids)):
        if number_of_times_encountered[i]>0:
            new_centroids[i]=new_centroids[i]/number_of_times_encountered[i]

    new_centroids=[ centroid.tolist() for centroid in new_centroids.values() if isinstance(centroid,np.ndarray) ]
    return new_centroids











