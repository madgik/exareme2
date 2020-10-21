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



    def all_in_one_algorithm(self, data_table, merged_local_results, parameters, attributes, result_table):
        init_centroids = []
        local_schema = "c1 FLOAT, c2 FLOAT"
        global_schema = "centroids STRING"
        iternum = 0
        for iternum in range(10):
            if iternum == 0:
                for i in range(parameters[0]):
                    init_centroids.append([random.randint(1, 100), random.randint(1, 100)])
                init_centroids = json.dumps(init_centroids)

                init_sqlscript = f'''
                            select * from
                            kmeans_local((select '{init_centroids}', {attributes[0]}, {attributes[1]} from {data_table}))
                            '''
                yield local_schema, init_sqlscript

            else:
                local_sqlscript = f'''
                                    select * from 
                                    kmeans_local((select centroids, {attributes[0]}, {attributes[1]} from {result_table},{data_table}))
                                    '''
                yield local_schema, local_sqlscript

            global_sqlscript = f'''
                    select kmeans_global({parameters[0]}, node_id, c1, c2) from {merged_local_results};
                    '''
            yield global_schema, global_sqlscript
            result = yield


## to test, you need one table in each local node containing 2 float attributes
## and send a pull request with 2 attributes and one int parameter (this param defines the number of clusters)
## or run sqlterm/mfederate.py and run `select kmeans(c1,c2,3) from data;`
## If using the existing servers in servers.py file there is already a table `data4`
## and kmeans can run with `select kmeans(c1,c2,3) from data4;`
