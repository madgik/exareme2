udf_list = []

numpy_sum = '''
CREATE or replace AGGREGATE numpy_sum(val BIGINT)
RETURNS BIGINT
LANGUAGE PYTHON {
    return numpy.sum(val)
};
'''
udf_list.append(numpy_sum)


numpy_count = '''
CREATE or replace AGGREGATE numpy_count(val BIGINT) 
RETURNS BIGINT
LANGUAGE PYTHON {
    import time
    time.sleep(4)
    return val.size
};
'''
udf_list.append(numpy_count)


pearson_local = '''
CREATE or replace FUNCTION pearson_local(val1 FLOAT, val2 FLOAT) 
RETURNS TABLE(sx FLOAT, sxx FLOAT, sxy FLOAT, sy FLOAT, syy FLOAT, n INT) 
LANGUAGE PYTHON {
    import sys
    sys.path.append("/home/openaire/monetdb_federated_poc/algorithms")
    import pearson_lib
    return pearson_lib.local(val1,val2)

};
'''
udf_list.append(pearson_local)


pearson_global = '''
CREATE or replace FUNCTION pearson_global(sx FLOAT, sxx FLOAT, sxy FLOAT, sy FLOAT, syy FLOAT, n INT) 
RETURNS TABLE(res FLOAT)
LANGUAGE PYTHON {
        import sys
        sys.path.append("/home/openaire/monetdb_federated_poc/algorithms")
        import pearson_lib
        return pearson_lib.merge(sx,sxx,sxy,sy,syy,n)
   
};
'''
udf_list.append(pearson_global)

kmeans_local = '''
CREATE or replace AGGREGATE kmeans_local(centroids JSON, column_x FLOAT, column_y FLOAT) 
RETURNS TABLE(c1 FLOAT, c2 FLOAT)
LANGUAGE PYTHON {
    import numpy as np
    import json
    from itertools import chain
    distances=[]
    centroids = json.loads(centroids[0])
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
    return np.array(new_centroids).T

};
'''
udf_list.append(kmeans_local)

kmeans_global = '''
CREATE or replace AGGREGATE kmeans_global(num_of_clusters INT, node_id INT, colx FLOAT, coly FLOAT) 
RETURNS STRING
LANGUAGE PYTHON {
    import numpy as np
    import json
    from itertools import chain
    
    #### transform monetdb columns to input of _global
    num_of_clusters = num_of_clusters[0]
    from collections import defaultdict
    d = defaultdict(list)
    li = []
    li.append(node_id)
    li.append(colx)
    li.append(coly)
    li2 = zip(*li)
    for k, n,p  in li2:
        d[k].append([n,p])
    local_centroids = []
    for i in d:
        local_centroids.append(d[i])
    #####################################
    
    if not local_centroids:
        new_centroids=[]
        for i in range(num_of_clusters):
            new_centroids.append([np.random.randint(100),np.random.randint(100)])
        return new_centroids

    if len(local_centroids)>1:

        num_of_centroids_returned_per_node=[ len(centroids) for centroids in local_centroids]

        m=max(num_of_centroids_returned_per_node)
        index_of_node_with_max_number_of_centroids=num_of_centroids_returned_per_node.index(max(num_of_centroids_returned_per_node))
        centroids=local_centroids[index_of_node_with_max_number_of_centroids]


        local_centroids.pop(index_of_node_with_max_number_of_centroids)
        data_points=local_centroids

        flatten=list(chain.from_iterable(data_points))
        colx=[x[0] for x in flatten]
        coly=[x[1] for x in flatten]

        def kmeans_step(centroids, column_x, column_y):
            # calculate distance between all datapoints(centroids,column_x,column_y) to all centroids
            distances = []
            data_points = np.array([column_x, column_y]).T
            for data_point in data_points:
                dist_row = []
                for centroid in centroids:
                    dist = abs(np.linalg.norm(data_point - centroid))
                    dist_row.append(dist)
                distances.append(dist_row)

            # assign each datapoint to the closest centroid (assign to clusters)
            cluster_assignment = []
            for data_point_distances in distances:
                min_index = np.where(data_point_distances == np.amin(data_point_distances))
                cluster_assignment.append(min_index[0][0])

            # calculate new centroids by calculatingt the mean of the datapoints assigned to each cluster
            new_centroids = {k: 0 for k in range(len(centroids))}
            number_of_times_encountered = {k: 0 for k in range(len(new_centroids))}
            for i in range(len(cluster_assignment)):
                new_centroids[cluster_assignment[i]] += data_points[i]
                number_of_times_encountered[cluster_assignment[i]] += 1
            # averaging them..
            for i in range(len(new_centroids)):
                if number_of_times_encountered[i] > 0:
                    new_centroids[i] = new_centroids[i] / number_of_times_encountered[i]

            new_centroids = [centroid.tolist() for centroid in new_centroids.values() if
                             isinstance(centroid, np.ndarray)]
            return new_centroids


        new_centroids=kmeans_step(centroids,colx,coly)

    else:
        flatten=list(chain.from_iterable(local_centroids))
        new_centroids=flatten
        colx=[x[0] for x in flatten]
        coly=[x[1] for x in flatten]

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

    return json.dumps(new_centroids)

};
'''
udf_list.append(kmeans_global)

#select * from kmeans_local((select lele.c1, data4.* from lele,data4));

#select kmeans_global(3,node_id,c1,c2) from fofo;