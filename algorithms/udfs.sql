CREATE or replace FUNCTION pearson_local(val1 FLOAT, val2 FLOAT)
RETURNS TABLE(sx FLOAT, sxx FLOAT, sxy FLOAT, sy FLOAT, syy FLOAT, n INT)
LANGUAGE PYTHON {
    import sys
    sys.path.append("/home/openaire/monetdb_federated_poc/algorithms")
    import pearson_lib
    return pearson_lib.local(val1,val2)

};
\

CREATE or replace AGGREGATE pearson_global(sx FLOAT, sxx FLOAT, sxy FLOAT, sy FLOAT, syy FLOAT, n INT)
RETURNS FLOAT
LANGUAGE PYTHON {
        import sys
        sys.path.append("/home/openaire/monetdb_federated_poc/algorithms")
        import pearson_lib
        return pearson_lib.merge(sx,sxx,sxy,sy,syy,n)

};
\


CREATE or replace AGGREGATE kmeans_local(centroids STRING, d_x FLOAT, d_y FLOAT)
RETURNS TABLE(col_x FLOAT, col_y FLOAT, points INT)
LANGUAGE PYTHON {
    distances=[]
    centroids = _conn.execute("select c_x,c_y from "+centroids[0]+ " ;")
    centroids = numpy.column_stack((centroids['c_x'],centroids['c_y']))

    for num, data_point in enumerate(d_x):
        dist_row=[]
        for centroid in centroids:
            dist=abs(numpy.linalg.norm((data_point,d_y[num])-centroid))
            dist_row.append(dist)
        distances.append(dist_row)

    #--assign each datapoint to the closest centroid (assign to clusters)
    cluster_assignment=[]
    for data_point_distances in distances:
        min_index=numpy.where(data_point_distances==numpy.amin(data_point_distances))
        cluster_assignment.append(min_index[0][0])

    #--sum x,y of points assigned to the same cluster, they will all be averaged in the global step
    to_return=numpy.zeros((3,len(centroids)))
    for i in range(len(cluster_assignment)):
        to_return[0][cluster_assignment[i]] += d_x[i]
        to_return[1][cluster_assignment[i]] += d_y[i]
        to_return[2][cluster_assignment[i]] += 1  #--this keeps track of the number of points assigned to this cluster

    return to_return
};
\

CREATE or replace AGGREGATE kmeans_global(num_of_clusters INT, colx FLOAT, coly FLOAT, points INT)
RETURNS TABLE(c_x FLOAT, c_y FLOAT)
LANGUAGE PYTHON {
    #--when kmeans starts the centroids (the centers of the clusters) will be randomly generated

    if type(points) == int and points == 0:
        new_centroids=numpy.random.randint(100, size=(2, num_of_clusters))
        return new_centroids

    num_of_clusters = num_of_clusters[0]
    #--averaging the centroids of the local nodes
    new_centroids=numpy.zeros((2,num_of_clusters))

    sums = []
    for cluster in range(num_of_clusters):
        sums.append(numpy.sum(points[(colx==colx[cluster]) & (coly==coly[cluster])]))
    i=0
    for row in sums:
        if row != 0:
            new_centroids[0][i] = colx[i]
            new_centroids[1][i] = coly[i]
        else:
            new_centroids[0][i] = numpy.random.randint(100)
            new_centroids[1][i] = numpy.random.randint(100)
        i+=1
    return new_centroids


};
\