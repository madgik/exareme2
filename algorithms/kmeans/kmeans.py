import numpy as np

def _local(centroids,data_points):
    '''
    parameters:
        centroids: numpy array of shape Cx2, where C the number of clusters
        data_points: numpy array of shape Dx2, where D the number of datapoints
    returns:
        a numpy array of shape Cx3, where C teh number of clusters. The 2 columns correspond
        to the sum of x and y coordinates of the datapoints assigned to each centroid and number of
        datapoints assigned to each centroids. These will be used in the global step to calculate averages 
    '''
    #calculate distance between all datapoints(centroids,column_x,column_y) to all centroids
    distances=[]
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

    #sum x,y of points assigned to the same cluster, they will all be averaged in the global step
    to_return=np.zeros((len(centroids),3))
    for i in range(len(cluster_assignment)):
        to_return[cluster_assignment[i]][0]+=data_points[i][0]
        to_return[cluster_assignment[i]][1]+=data_points[i][1]
        to_return[cluster_assignment[i]][2]+=1#this keeps track of the number of points assigned to this cluster

    return to_return

def _global(num_of_clusters,local_centroids=None):
    '''
    parameters:
        num_of_clusters: the number of desired clusters
        local_centroids: numpy array of shape NxCx3 where N:number of Nodes, C:number of Clusters. The last
                        size 3 corresponds to x,y,number of occurences
    returns:
        the new centroids
    '''
    #when kmeans starts the centroids (the centers of the clusters) will be randomly generated
    if not local_centroids:
        new_centroids=np.random.randint(100, size=(num_of_clusters, 2))
        return new_centroids

    #averaging the centroids of the local nodes
    new_centroids=np.zeros((num_of_clusters,2))
    summed=np.sum(local_centroids,axis=0)
    i=0
    for row in summed:
        if row[2]!=0:
            new_centroids[i]=[row[0]/row[2],row[1]/row[2]]
        else:
            new_centroids[i]=np.random.randint(100, size=(1, 2))
        i+=1
    return new_centroids


