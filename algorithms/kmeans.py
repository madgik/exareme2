# Two different implementations follow: The difference is that the first implements the iteration condition in the database, the second in the dataflow level
# Default is the second. There are 2 udfs, designed so that they are reusable: a udf that calculates the euclideian distance between 2 points (in any dimension)
# and a udf that calculates a random integer in a given range

udf_list = []
udf_list.append('''
CREATE OR REPLACE FUNCTION EUCLIDEAN_DISTANCE(*)
RETURNS FLOAT LANGUAGE PYTHON {
   # calculates and returns the euclideian distance between 2 points
   # if there are N arguments the first N/2 refer to dimensions of the first point 
   # and the rest refers to the corresponding dimensions of the second point
   # example usage `select euclidean distance(x1,y1,x2,y2) from datax,datay`;
   sums = 0.0
   for i in range(1,int(len(_columns)/2+1)):
       sums += numpy.power(_columns['arg'+str(i)]-_columns['arg'+str(int(i+len(_columns)/2))],2)
   return numpy.sqrt(sums)
};
''')

udf_list.append('''
# calculates a random integer in the given range (num1, num2)
# example usage `select random(1,100);`
CREATE OR REPLACE FUNCTION RANDOM(num integer, num2 integer)
RETURNS FLOAT LANGUAGE PYTHON{
   return numpy.random.randint(num,num2)
};
''')

class Algorithm2: # full sql version
    def algorithm(self, data_table, merged_local_results, parameters, attributes, result_table):
        for iternum in range(200):
            yield self._global(iternum, merged_local_results, parameters, attributes, result_table)
            yield self._local(iternum, data_table, parameters, attributes, result_table)

    def _local(self, iternum, data_table, parameters, attributes, result_table):
        schema = "N INT, centx FLOAT, centy FLOAT, datax FLOAT, datay FLOAT"
        
        sqlscript = f'''
            select count(*) as N, centx, centy, sum(datax) as datax, sum(datay) as datay from
                (
                select row_number() over (
                                          partition by {attributes[0]}, {attributes[1]} 
                                          order by EUCLIDEAN_DISTANCE({attributes[0]}, {attributes[1]},centx,centy)
                                         ) as id, 
                        {attributes[0]} as datax,
                        {attributes[1]} as datay,
                        centx, 
                        centy
                from {data_table}, (select * from {result_table} where iternum = {iternum}) centroids
                ) X where id=1 group by centx, centy
        '''
        return schema, sqlscript, 'local'


    def _global(self, iternum, merged_local_results, parameters, attributes, result_table):
        schema = "termination BOOL, iternum INT,  centx FLOAT, centy FLOAT"
        if iternum == 0:
            sqlscript = f'''
                 select false, 0, random(value)  as centx, random(value+1) as centy from generate_series(0,{parameters[0]})
            '''
        else:
        	sqlscript = f'''
        	select not exists (select centx, centy  except select centx,centy from {result_table} where iternum = {iternum-1}) as termination, {iternum}, centx, centy from (
        	select 
        	centx, 
        	centy from 
             	(select sum(datax)/sum(n) as centx, 
                 	    sum(datay)/sum(n) as centy  
                    	from {merged_local_results} group by centx, centy
             	) T   	
        	union all
        	
        	select random(value) as centx, random(value+1) as centy 
        	    from generate_series(0, {parameters[0]} - (select count(*) from (select distinct centx, centy from {merged_local_results}) X) )
        	) result order by termination desc;
        	'''
        return schema, sqlscript, 'global'


class Algorithm: # iteration condition in python
    def algorithm(self, data_table, merged_local_results, parameters, attributes, result_table):
        centroids = []
        for iternum in range(10000):
            yield self._global(iternum, merged_local_results, parameters, attributes, result_table)
            new_centroids = yield
            if new_centroids == centroids:
                break
            else:
                centroids = new_centroids
            yield self._local(iternum, data_table, parameters, attributes, result_table)


    def _local(self, iternum, data_table, parameters, attributes, result_table):
        schema = "N INT, centx FLOAT, centy FLOAT, datax FLOAT, datay FLOAT"

        sqlscript = f'''
            select count(*) as N, centx, centy, sum(datax) as datax, sum(datay) as datay from
                (
                select row_number() over (
                                          partition by {attributes[0]}, {attributes[1]} 
                                          order by EUCLIDEAN_DISTANCE({attributes[0]}, {attributes[1]},centx,centy)
                                         ) as id, 
                        {attributes[0]} as datax,
                        {attributes[1]} as datay,
                        centx, 
                        centy
                from {data_table}, {result_table}
                ) X where id=1 group by centx, centy
        '''
        return schema, sqlscript, 'local'

    def _global(self, iternum, merged_local_results, parameters, attributes, result_table):
        schema = "centx FLOAT, centy FLOAT"
        if iternum == 0:
            sqlscript = f'''
                 select random(0,100) as centx, random(0,100) as centy from generate_series(0,{parameters[0]})
            '''
        else:
            sqlscript = f'''
            select sum(datax)/sum(n), sum(datay)/sum(n) from {merged_local_results} group by centx, centy
        	union all
        	select random(0,100), random(0,100) from 
        	    generate_series(0, {parameters[0]} - 
        	                       (select count(*) from (select distinct centx, centy from {merged_local_results}) X) )
        	'''
        return schema, sqlscript, 'global'
