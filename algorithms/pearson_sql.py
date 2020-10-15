## 5% faster than with numpy UDFs in a dataset which contains 5 rows (3 in one local and 2 in the other).
class Algorithm:
    def __init__(self):
        pass

    ### this function is not in the current execution flow - if it is, it works as an input to run_algorithm.dataflow_parse_and_execute
    def algorithm(self, viewlocaltable,globaltable,  parameters, attributes, globalresulttable):
        iternum = 0
        yield self._local(iternum, viewlocaltable, parameters, attributes, globalresulttable)
        yield self._global(iternum, globaltable, parameters, attributes)

    def _local(self, iternum, viewlocaltable, parameters, attributes, globalresulttable):
        #### todo convert schema to a list and not string
        schema = "sx FLOAT, sxx FLOAT, sxy FLOAT, sy FLOAT, syy FLOAT, n INT"
        sqlscript = f'''
        SELECT
            SUM(x) as sx, 
            SUM(x*x) as sxx, 
            SUM(x*y) as sxy, 
            SUM(y) as sy, SUM(y*y) as syy, 
            COUNT(x) as n 
        FROM (
                SELECT 
                    {attributes[0]} as x,
                    {attributes[1]} as y 
                FROM {viewlocaltable}
             )  pearson_data;
        '''

        f'''
        
            SELECT 
                SUM(x) as x,
                SUM(xx) as sxx,
                SUM(xy) as sxy
                SUM(y) as sy,
                SUM(yy) as syy,
                COUNT(x) as n
            FROM
                    (SELECT
                        pow(x,2) as xx, 
                        x*y as xy, 
                        pow(y,2) as syy, 
                        COUNT(x) as n 
                    FROM (
                            SELECT 
                                {attributes[0]} as x,
                                {attributes[1]} as y 
                            FROM {viewlocaltable}
                         )  pearson_data
                    ) operations;
                    '''

        return schema, sqlscript


    def _global(self, iternum, globaltable, parameters, attributes):
        #### todo convert schema to a list and not string
        schema = "result FLOAT"
        sqlscript  = f'''
        SELECT  
            CAST((n * sxy - sx * sy) AS float)/
            (SQRT(n * sxx - sx * sx) * SQRT(n * syy - sy * sy))
            AS result
        FROM (
                SELECT SUM(n) as n,
                       SUM(sx) as sx,
                       SUM(sxx) as sxx,
                       SUM(sxy) as sxy,
                       SUM(sy) as sy,
                       SUM(syy) as syy 
                FROM {globaltable} 
             )  pearson_sums;
        '''
        return schema, sqlscript


## select pearson_global(SUM(sx),SUM(sxx),SUM(sxy),SUM(sy),SUM(syy),SUM(n)) from globaltable;
