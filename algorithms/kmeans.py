import random
import json

class Algorithm:

    def algorithm(self, data_table, merged_local_results, parameters, attributes, result_table):
        iternum = 0
        for iternum in range(10):
            yield self._local(iternum, data_table, parameters, attributes, result_table)
            yield self._global(iternum, merged_local_results, parameters, attributes)
            res = yield

    def _local(self, iternum, data_table, parameters, attributes, result_table):
        #### todo convert schema to a list and not string
        schema = "c1 FLOAT, c2 FLOAT"
        if iternum == 0:
            new_centroids = []
            for i in range(parameters[0]):
                new_centroids.append([random.randint(1,100), random.randint(1,100)])
            init_centroids = json.dumps(new_centroids)

            sqlscript = f'''
            select * from
            kmeans_local((select '{init_centroids}', {attributes[0]}, {attributes[1]} from {data_table}))
            '''

        else:
            sqlscript = f'''
            select * from 
                kmeans_local((select centroids, {attributes[0]}, {attributes[1]} from {result_table},{data_table}))
            '''
        return schema, sqlscript


    def _global(self, iternum, merged_local_results, parameters, attributes):
        #### todo convert schema to a list and not string
        schema = "centroids STRING"
        sqlscript  = f'''
        select kmeans_global({parameters[0]}, node_id, c1, c2) from {merged_local_results};
        '''
        return schema, sqlscript
