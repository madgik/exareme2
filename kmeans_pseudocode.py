def kmeans(num_of_clusters, data_tables_names, data_nodes,num_of_iterations, global_node){
    '''
    kmeans clustering:
        1. assign each datapoint to the closer centroid 
        2. calculate new centroids based on the set of datapoints assigned to each of the previous centroids
        3. repeat from setp 1, until some condition (simplest a predefined number of iterations)

    kmeans clustering(simple distributed version):
        1. GLOBAL node: 
            1.1. generates an initial (random) set of centroids
            1.2. makes centroids(table) visible to all local nodes 
        2. LOCAL nodes:
            2.1. find which centroid each datapoint is closer to 
            2.2. calculate new centroids based on the datapoints assigned to each of the previous centroids
            2.3. make new (local)centroids(table) visible to the GLOBAL node
        3. GLOBAL node:
            3.1. calculates new centroids by averaging the local centroids  
            3.2. makes centroids(table) visible to all local nodes 
        4. repeat from step 2, until some condition (here num_of_iterations)


    Parameters:
    num_off_clusters: the num of clusters the k-means must produce
    datanodes: dictionary in the form   {   database1:data_table,
                                            database2:data_table,
                                            ...
                                        }
    data_tables_names: the name of the tables that contain the actual datapoints, (most likely VIEW tables)
    data_nodes: the nodes/databases that contain the data_tables_names
    '''

    params={
            "num_of_rows":num_of_clusters
            "num_of_cols": #function returning dimension of a table??
            "variable_type":float
            "range":#some range..
            }
    centroids= udfs.generate_random( params, global_node, data_nodes )
    
    counter=0
    while(counter<num_of_iterations){
        points_assigned_to_clusters=[]
        recalculated_centroids=[]
        for data_node in data_nodes:
            #calculate distances each datapoint from each centroid
            distances_from_centroids= udfs_wrapper.calculate_norms( centroids, data_nodes[data_node], data_node, [] )#result table visible only to local node

            #find the minimum distance centroid for each datapoint
            #min_column udf will return only the indices 
            tmp=udfs_wrapper.min_column( distances_from_centroids, data_node, global_node ) #result table visible to local AND global node
            datapoints_assigned_to_clusters.append(tmp)

            #calculate new centroids
            tmp=udfs_wrapper.means_by_index( points_assigned_to_clusters, data_nodes[data_node], data_node, global_node)#result table visible to local AND global node
            recalculated_centroids.append(tmp)

        merged_datapoints_assigned_to_clusters="merged_datapoints_assigned_to_clusters"
        schema=#some schema...
        create_merge_table(global_node,merged_datapoints_assigned_to_clusters,schema,datapoints_assigned_to_clusters)

        merged_local_recalculated_centroids="merged_recalculated_centroids"
        schema=#some schema...
        create_merge_table(global_node,merged_recalculated_centroids,schema,recalculated_centroids)
        #calculate new centroids

        centroids=udfs_wrapper.means_by_index(merged_datapoints_assigned_to_clusters,merged_recalculated_centroids,global_node,data_nodes)#result table visible to global node AND ALL local nodes

        counter++
    }

    return numpy.array(centroids)






