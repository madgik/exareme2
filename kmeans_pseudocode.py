def kmeans(num_of_clusters, num_of_iterations, data_dbs, global_db){
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
    num_off_clusters:   the num of clusters the k-means must produce
    data_dbs:   dictionary in the form   {   database1:data_table,
                                            database2:data_table,
                                            ...
                                        }
                where database1,database2,... connection object to the respective dbs
                
    global_db:  a connection object to the global database
    '''

    params={
            "num_of_rows":num_of_clusters
            "num_of_cols": #function returning dimension of a table??
            "variable_type":float
            "range":#some range..
            }
    centroids= udfs.generate_random( params, global_db, data_dbs )
    
    counter=0
    while(counter<num_of_iterations){
        points_assigned_to_clusters=[]
        recalculated_centroids=[]
        for local_db in data_dbs:
            #calculate distance for each datapoint to each centroid
            distances_from_centroids= udfs_wrapper.calculate_norms( centroids, data_dbs[local_db], local_db, [] )#result table visible only to local node

            #find the minimum distance centroid for each datapoint
            #min_column udf will return the index of the corresponding centroid
            tmp=udfs_wrapper.min_column( distances_from_centroids, local_db, global_db ) #result table visible to local AND global node
            datapoints_assigned_to_clusters.append(tmp)

            #calculate new centroids
            tmp=udfs_wrapper.means_by_index( tmp, data_nodes[database], local_db, global_db)#result table visible to local AND global node
            recalculated_centroids.append(tmp)

        merged_datapoints_assigned_to_clusters="merged_datapoints_assigned_to_clusters"
        schema=#some schema...
        create_merge_table(global_db,merged_datapoints_assigned_to_clusters,schema,datapoints_assigned_to_clusters)

        merged_local_recalculated_centroids="merged_recalculated_centroids"
        schema=#some schema...
        create_merge_table(global_db,merged_recalculated_centroids,schema,recalculated_centroids)
        #calculate new centroids

        centroids=udfs_wrapper.means_by_index(merged_datapoints_assigned_to_clusters,merged_recalculated_centroids,global_db,data_dbs)#result table visible to global node AND ALL local nodes

        counter++
    }

    return numpy.array(centroids)






