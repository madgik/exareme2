class Algorithm:
    def __init__(self):
        pass

    ### this function is not in the current execution flow - if it is, it works as an input to run_algorithm.dataflow_parse_and_execute
    def dataflow(self, viewlocaltable, parameters, globaltable, attributes, globalresulttable):
        iternum = 0
        self._local(iternum, viewlocaltable, parameters, attributes, globalresulttable)
        return self._global(iternum, globaltable, parameters, attributes)


    def _local(self, iternum, viewlocaltable, parameters, attributes, globalresulttable):
        #### todo convert schema to a list and not string
        schema = "sx FLOAT, sxx FLOAT, sxy FLOAT, sy FLOAT, syy FLOAT, n INT"
        sqlscript = f'''
        SELECT
            sum(x) as sx, 
            sum(x*x) as sxx, 
            sum(x*y) as sxy, 
            sum(y) as sy, sum(y*y) as syy, 
            count(x) as n 
        FROM (
                SELECT 
                    {attributes[0]} as x, 
                    {attributes[1]} as y 
                FROM {viewlocaltable}
             )  pearson_data;
        '''


        return schema, sqlscript


    def _global(self, iternum, globaltable, parameters, attributes):
        #### todo convert schema to a list and not string
        schema = "result FLOAT"
        sqlscript  = f'''
        SELECT  
            CAST((n * sxy - sx * sy) AS float)/
            (sqrt(n * sxx - sx * sx) * sqrt(n * syy - sy * sy))
        FROM (
                SELECT sum(n) as n,
                       sum(sx) as sx,
                       sum(sxx) as sxx,
                       sum(sxy) as sxy,
                       sum(sy) as sy,
                       sum(syy) as syy 
                FROM {globaltable} 
             )  pearson_sums;
        '''
        return schema, sqlscript


## select pearson_global(sum(sx),sum(sxx),sum(sxy),sum(sy),sum(syy),sum(n)) from globaltable;
