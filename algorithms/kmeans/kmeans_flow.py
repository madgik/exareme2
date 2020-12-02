from run_algorithm import AlgorithmFlow_abstract


#DEBUG
import pdb

class AlgorithmFlow(AlgorithmFlow_abstract):

    async def run(self):
        runtime_interface=self.runtime_interface
        print(f"(kmeans_my::AlgorithmFlow::run) algorithm_parameters->{self.algorithm_parameters}")

        #print(f'(kmeans_my::AlgorithmFlow::run) calling->  self.runtime_interface.execute_udf_on_global_node("kmeans._global",num_of_clusters,None)')
        num_of_clusters=self.algorithm_parameters["numOfClusters"]
        num_of_iterations=self.algorithm_parameters["numOfIterations"]
        initial_min=self.algorithm_parameters["initialMin"]
        initial_max=self.algorithm_parameters["initialMax"]

        #TODO: should somewhere do some check that the data table schema is consistent along all local nodes?
        num_of_dimensions=await self.runtime_interface.get_number_of_columns(self.data_table_name,self.runtime_interface._nodes["local"][0])
        print(f"(kmeans_my::AlgorithmFlow::run) num_of_dimensions-> {num_of_dimensions} ")

        #TODO:the result schema should not be explicitly defined here...
        udf_result_schema=[ f"centroid_d{i} DOUBLE" for i in range(num_of_dimensions)]
        udf_result_schema=",".join(udf_result_schema) #-->"centroid_d1 DOUBLE, centroid_d2 DOUBLE, ..."
        print(f'(kmeans_my::AlgorithmFlow::run) calling->  self.runtime_interface.execute_udf_on_global_node({"generate_initial_centroids"},{udf_result_schema},{[num_of_dimensions,num_of_clusters,initial_min,initial_max]}')
        centroids_table_name=await  self.runtime_interface.execute_udf_on_global_node("generate_initial_centroids",udf_result_schema,[num_of_dimensions,num_of_clusters,initial_min,initial_max])
        pdb.set_trace()

#async def execute_udf(self,udf_name,udf_result_schema,input_table_name):#more than one input tables?        
#def _global(num_of_dimensions:int,num_of_clusters:int,local_centroids:numpy.ndarray=None)->numpy.ndarray:

        centroids_table_name=await  self.runtime_interface.execute_udf_on_global_node("_global",[num_of_dimensions,num_of_clusters,None])
        #async def execute_udf(self,udf_name,input_args,return_args):#more than one input tables?
###        for i in range(1,10):

###            local_centroids=await self.runtime_interface.execute_udf_on_all_local_nodes("kmeans._local",result_table_schema,self.data_table_name)

###            merged_table_name=await self.runtime_interface.create_remote_and_merge_on_global_node(local_centroids):

###            centroids_table_name=await  self.runtime_interface.execute_udf_on_global_node("kmeans._global",num_of_clusters,merged_table_name)
###            create_remote_on_locals
###            
        result_table_schema= "sx FLOAT, sxx FLOAT, sxy FLOAT, sy FLOAT, syy FLOAT, n INT"
        #TODO:async stuff should not(?) be on the alg dev?        
        result1=await self.runtime_interface.execute_udf_on_all_local_nodes("local_pearson",result_table_schema,self.data_table_name)
        print(f"(kmeans_my::AlgorithmFlow::run) just after local_pearson result1->{result1}") 

        result_table_schema = "result FLOAT"
        result2=await self.runtime_interface.execute_udf_on_global_node("global_pearson",result_table_schema,result1)
        print(f"(kmeans_my::AlgorithmFlow::run) just after global_pearson result2->{result2}") 

        return result2


#nodes_aliases=orchestrator.get_nodes_aliases()
#global_node_alias=nodes_aliases["global"]
#local_nodes_aliases=nodes_aliases["local"]

#create_data_views_on_all_local_nodes()

#---------------
#def _global(num_of_clusters,local_centroids=None)
#def _local(centroids,data_points)
