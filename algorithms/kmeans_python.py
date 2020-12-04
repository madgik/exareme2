udf_list = []
udf_list.append('''
CREATE OR REPLACE FUNCTION EUCLIDEAN_DISTANCE(*)
RETURNS FLOAT LANGUAGE PYTHON {
   sums = 0.0
   for i in range(1,int(len(_columns)/2+1)):
       sums += numpy.power(_columns['arg'+str(i)]-_columns['arg'+str(int(i+len(_columns)/2))],2)
   return numpy.sqrt(sums)
};
''')

class Algorithm: # iteration condition in python
    def algorithm(self, data_table, merged_local_results, parameters, attributes, result_table):
        # init schemata yield
        yield ["N INT, centx FLOAT, centy FLOAT, datax FLOAT, datay FLOAT", "centx FLOAT, centy FLOAT", "schema"]
        centroids = []
        for iternum in range(200):
            yield self.global_aggregation(merged_local_results, parameters)
            new_centroids = yield
            if new_centroids != centroids: centroids = new_centroids
            else: break
            yield self.local_expectation(data_table, attributes, result_table)

    def local_expectation(self, data_table, attr, result_table):
        sqlscript = f'''
            select count(*) as N, centx, centy, sum(datax) as datax, sum(datay) as datay from
                (
                select row_number() over (
                                          partition by datax, datay 
                                          order by EUCLIDEAN_DISTANCE(datax, datay ,centx, centy)
                                         ) as id, datax, datay, centx, centy
                from (select {attr[0]} as datax, {attr[1]} as datay from {data_table}) as data_points, 
                     {result_table} as centroids
                ) expectations where id=1 group by centx, centy
        '''
        return [sqlscript, 'local']

    def global_aggregation(self, merged_local_results, parameters):
        centroids_n = parameters[0]
        sqlscript = f'''
        select cent_x, cent_y from (
            select sum(n) as points, sum(datax)/sum(n) as cent_x, sum(datay)/sum(n) as cent_y from {merged_local_results} group by centx, centy
        	union all
        	select 0, rand()%2+2, rand()%2+2 from generate_series(0, {centroids_n})
            order by points desc limit {centroids_n}
            ) global_centroids
        	'''
        return [sqlscript, 'global']